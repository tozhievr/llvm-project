#!/usr/bin/env python3
"""Build the freestanding libc++26 modules for Windows x64 with Clang.

This is the first native-Python build target for the libc++26 migration. It
discovers named-module dependencies, generates the target-feature module,
compiles module interfaces and the native runtime, and archives the objects.
It deliberately does not invoke CMake, MSBuild, the Windows SDK, or a C runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


MODULE_DECLARATION = re.compile(
    r"^\s*(?:export\s+)?module\s+"
    r"([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*)\s*;",
    re.MULTILINE,
)
MODULE_IMPORT = re.compile(
    r"^\s*(?:export\s+)?import\s+"
    r"([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*)\s*;",
    re.MULTILINE,
)
BUILD_FEATURES = ("SSE42", "AVX2", "FMA")
BUILD_TARGET = "x86_64-pc-windows-msvc"


@dataclass(frozen=True)
class ModuleUnit:
    name: str
    path: Optional[Path]
    source: Optional[str]
    imports: Tuple[str, ...]


def configuration_module() -> str:
    enabled = set(BUILD_FEATURES)
    feature_types = (
        "SSE42",
        "AVX2",
        "AVX512",
        "AVX512BW",
        "FMA",
        "VNNI",
    )
    specializations = "\n".join(
        "template <> inline constexpr bool kIsEnabledFeature<%s> = %s;"
        % (feature, "true" if feature in enabled else "false")
        for feature in feature_types
    )
    declarations = "\n".join("struct %s {};" % feature for feature in feature_types)
    return """\
export module cmake_generated;

export namespace cmake_generated {
struct PlatformWindows {};
struct PlatformLinux {};
struct PlatformApple {};
struct EndianessLittle {};
struct EndianessBig {};
%s
inline constexpr bool kIsWindowsPlatform = true;
inline constexpr bool kIsLinuxPlatform = false;
inline constexpr bool kIsApplePlatform = false;
inline constexpr bool kIsLittleEndian = true;
inline constexpr bool kIsBigEndian = false;
template <class Feature>
inline constexpr bool kIsEnabledFeature = false;
%s
}  // namespace cmake_generated
""" % (declarations, specializations)


def _read_module(name: str, path: Path) -> ModuleUnit:
    text = path.read_text(encoding="utf-8-sig")
    declarations = MODULE_DECLARATION.findall(text)
    if len(declarations) != 1:
        raise ValueError(
            "%s: expected exactly one named module declaration, found %d"
            % (path, len(declarations))
        )
    module_name = declarations[0]
    if ":" in module_name:
        raise ValueError("%s: module partitions are not supported" % path)
    return ModuleUnit(module_name, path, None, tuple(MODULE_IMPORT.findall(text)))


def discover_modules(source_root: Path) -> Dict[str, ModuleUnit]:
    modules: Dict[str, ModuleUnit] = {}
    generated = configuration_module()
    modules["cmake_generated"] = ModuleUnit(
        "cmake_generated",
        None,
        generated,
        tuple(MODULE_IMPORT.findall(generated)),
    )

    for path in sorted(source_root.rglob("*.cppm")):
        module = _read_module(path.stem, path)
        if module.name in modules:
            raise ValueError(
                "duplicate module %s: %s and %s"
                % (module.name, modules[module.name].path, path)
            )
        modules[module.name] = module

    if len(modules) == 1:
        raise ValueError("no C++ module interfaces found under %s" % source_root)

    for module in modules.values():
        missing = sorted(set(module.imports) - set(modules))
        if missing:
            location = str(module.path) if module.path else "<generated config>"
            raise ValueError(
                "%s: imported module(s) not found: %s"
                % (location, ", ".join(missing))
            )
    return modules


def topological_order(modules: Dict[str, ModuleUnit]) -> List[ModuleUnit]:
    ordered: List[ModuleUnit] = []
    active: List[str] = []
    visited: Set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in active:
            cycle = active[active.index(name) :] + [name]
            raise ValueError("module import cycle: " + " -> ".join(cycle))
        active.append(name)
        for dependency in modules[name].imports:
            visit(dependency)
        active.pop()
        visited.add(name)
        ordered.append(modules[name])

    for name in sorted(modules):
        visit(name)
    return ordered


def _signature(command: Sequence[str], inputs: Iterable[Path]) -> str:
    records = []
    for path in inputs:
        stat = path.stat()
        records.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(
        {"command": list(command), "inputs": records},
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload


def _needs_build(
    outputs: Sequence[Path],
    stamp: Path,
    command: Sequence[str],
    inputs: Iterable[Path],
    force: bool,
) -> Tuple[bool, str]:
    input_list = tuple(inputs)
    current = _signature(command, input_list)
    if force or not stamp.is_file() or not all(path.is_file() for path in outputs):
        return True, current
    try:
        return stamp.read_text(encoding="utf-8") != current, current
    except OSError:
        return True, current


def _write_stamp(stamp: Path, signature: str) -> None:
    stamp.parent.mkdir(parents=True, exist_ok=True)
    temporary = stamp.with_suffix(stamp.suffix + ".tmp")
    temporary.write_text(signature, encoding="utf-8")
    temporary.replace(stamp)


def _run(
    command: Sequence[str],
    stamp: Path,
    outputs: Sequence[Path],
    inputs: Iterable[Path],
    force: bool,
    dry_run: bool,
) -> bool:
    if dry_run:
        print("+ %s" % subprocess.list2cmdline(list(command)))
        return True
    rebuild, signature = _needs_build(outputs, stamp, command, inputs, force)
    if not rebuild:
        print("up to date: %s" % outputs[0])
        return False
    print("building: %s" % outputs[0])
    subprocess.run(command, check=True)
    _write_stamp(stamp, signature)
    return True


def _base_flags() -> List[str]:
    return [
        "--target=" + BUILD_TARGET,
        "-std=c++26",
        "-march=x86-64-v3",
        "-mavx2",
        "-mfma",
        "-ffreestanding",
        "-fno-exceptions",
        "-fno-rtti",
        "-fno-stack-protector",
        "-fno-builtin",
        "-nostdinc",
        "-O2",
    ]


def build(
    source_root: Path,
    build_root: Path,
    compiler: str,
    archiver: str,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    source_root = source_root.resolve(strict=True)
    build_root = build_root.resolve()
    modules = discover_modules(source_root)
    ordered = topological_order(modules)
    flags = _base_flags()
    module_search_path = "-fprebuilt-module-path=%s" % (build_root / "modules")
    pcm_paths = {module.name: build_root / "modules" / (module.name + ".pcm") for module in ordered}
    object_paths: List[Path] = []
    available_modules: List[str] = []
    changed = False

    for module in ordered:
        pcm = pcm_paths[module.name]
        obj = build_root / "objects" / (module.name + ".obj")
        if not dry_run:
            pcm.parent.mkdir(parents=True, exist_ok=True)
            obj.parent.mkdir(parents=True, exist_ok=True)
        source = module.path
        if source is None:
            source = build_root / "generated" / "cmake_generated.cppm"
            if not dry_run:
                source.parent.mkdir(parents=True, exist_ok=True)
                if not source.exists() or source.read_text(encoding="utf-8") != module.source:
                    source.write_text(module.source or "", encoding="utf-8")
        assert source is not None
        dependencies = [pcm_paths[name] for name in available_modules]
        module_flags = [module_search_path]
        precompile_command = [
            compiler,
            *flags,
            *module_flags,
            "--precompile",
            str(source),
            "-o",
            str(pcm),
        ]
        changed |= _run(
            precompile_command,
            pcm.with_suffix(".pcm.stamp"),
            [pcm],
            [source, *dependencies],
            force,
            dry_run,
        )

        object_command = [
            compiler,
            *flags,
            *module_flags,
            "-c",
            str(pcm),
            "-o",
            str(obj),
        ]
        changed |= _run(
            object_command,
            obj.with_suffix(".obj.stamp"),
            [obj],
            [pcm, *dependencies],
            force,
            dry_run,
        )
        object_paths.append(obj)
        available_modules.append(module.name)

    runtime_source = source_root / "winnt_core" / "libcpp26.winnt.runtime.cpp"
    if not runtime_source.is_file():
        raise ValueError("required native runtime source not found: %s" % runtime_source)
    runtime_obj = build_root / "objects" / "libcpp26.winnt.runtime.obj"
    runtime_command = [
        compiler,
        *flags,
        "-c",
        str(runtime_source),
        "-o",
        str(runtime_obj),
    ]
    changed |= _run(
        runtime_command,
        runtime_obj.with_suffix(".obj.stamp"),
        [runtime_obj],
        [runtime_source],
        force,
        dry_run,
    )
    object_paths.append(runtime_obj)

    archive = build_root / "lib" / "libcpp26.lib"
    archive_command = [archiver, "rcD", str(archive), *(str(path) for path in object_paths)]
    archive_stamp = archive.with_suffix(".lib.stamp")
    if dry_run:
        archive_rebuild = True
        archive_signature = ""
    else:
        archive_signature = _signature(archive_command, object_paths)
        archive_rebuild = (
            force
            or changed
            or not archive.is_file()
            or not archive_stamp.is_file()
            or archive_stamp.read_text(encoding="utf-8") != archive_signature
        )
    if archive_rebuild:
        if dry_run:
            print("+ %s" % subprocess.list2cmdline(archive_command))
        if not dry_run:
            print("archiving: %s" % archive)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.unlink(missing_ok=True)
            subprocess.run(archive_command, check=True)
            _write_stamp(archive_stamp, archive_signature)
    else:
        print("up to date: %s" % archive)
    return archive


def _tool(value: Optional[str], name: str) -> str:
    if value:
        return value
    found = shutil.which(name)
    if not found:
        raise ValueError("%s was not found on PATH; pass --%s explicitly" % (name, name))
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root",
        type=Path,
        required=True,
        help="root of the libc++26 source tree",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build") / "libcpp26",
        help="output directory (default: build/libcpp26)",
    )
    parser.add_argument("--clang", help="clang++ executable (default: clang++ on PATH)")
    parser.add_argument("--llvm-ar", help="llvm-ar executable (default: llvm-ar on PATH)")
    parser.add_argument("--force", action="store_true", help="rebuild all outputs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print compiler and archiver commands without executing them",
    )
    args = parser.parse_args(argv)
    try:
        clang = _tool(args.clang, "clang++")
        llvm_ar = _tool(args.llvm_ar, "llvm-ar")
        archive = build(
            args.library_root,
            args.build_dir,
            clang,
            llvm_ar,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print("build failed: %s" % error, file=sys.stderr)
        return 1
    if not args.dry_run:
        print("built: %s" % archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
