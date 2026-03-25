import os
import tempfile
from typing import Optional
import sys
from bioio import BioImage

import tifffile
import numpy as np
from numpy.typing import NDArray
from PIL import Image


# TODO: Add a try catch here
# We only need this if we are using bioformats, and we need to have Java
import jdk4py
import scyjava


BIOIO_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".nd2", ".lif", ".czi", ".dv", ".ims", ".oib", ".oif"}
)


_BIOIMAGE_CACHE_PATH: str | None = None
_BIOIMAGE_CACHE_IMAGE: BioImage | None = None


def _normalize_to_uint8(arr: NDArray) -> NDArray[np.uint8]:
    arr = arr.astype(np.float64)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if np.isnan(min_val) or np.isnan(max_val) or max_val - min_val == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    normalized = (arr - min_val) / (max_val - min_val) * 255
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _cache_dir_for(path: str) -> str:
    base, _ = os.path.splitext(path)
    cache_dir = f"{base}.labelme_cache"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _cache_file_for(
    path: str,
    *,
    t_index: int,
    z_index: int,
    channel_indices: list[int],
) -> str:
    channel_key = "none" if not channel_indices else "-".join(str(c) for c in channel_indices)
    filename = f"t{t_index:04d}_z{z_index:04d}_c{channel_key}.png"
    return os.path.join(_cache_dir_for(path), filename)


def _get_bioimage(path: str) -> BioImage:
    global _BIOIMAGE_CACHE_IMAGE
    global _BIOIMAGE_CACHE_PATH
    if _BIOIMAGE_CACHE_PATH == path and _BIOIMAGE_CACHE_IMAGE is not None:
        return _BIOIMAGE_CACHE_IMAGE
    _BIOIMAGE_CACHE_IMAGE = load_tczyx_image(path)
    _BIOIMAGE_CACHE_PATH = path
    return _BIOIMAGE_CACHE_IMAGE


def is_bioio_stack_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in BIOIO_IMAGE_SUFFIXES


def get_bioio_stack_info(path: str) -> dict[str, object]:
    img = _get_bioimage(path)
    size_t = max(1, int(getattr(img.dims, "T", 1)))
    size_z = max(1, int(getattr(img.dims, "Z", 1)))
    size_c = max(1, int(getattr(img.dims, "C", 1)))

    channel_names_raw = getattr(img, "channel_names", None)
    if isinstance(channel_names_raw, list) and len(channel_names_raw) == size_c:
        channel_names = [str(name) for name in channel_names_raw]
    else:
        channel_names = [f"C{idx}" for idx in range(size_c)]

    return {
        "size_t": size_t,
        "size_z": size_z,
        "size_c": size_c,
        "channel_names": channel_names,
    }


def render_bioio_composite_png(
    path: str,
    *,
    t_index: int = 0,
    z_index: int = 0,
    channel_indices: list[int] | None = None,
) -> str:
    img = _get_bioimage(path)

    size_t = max(1, int(getattr(img.dims, "T", 1)))
    size_z = max(1, int(getattr(img.dims, "Z", 1)))
    size_c = max(1, int(getattr(img.dims, "C", 1)))

    t = max(0, min(t_index, size_t - 1))
    z = max(0, min(z_index, size_z - 1))

    if channel_indices is None:
        selected_channels = list(range(min(size_c, 3)))
    else:
        selected_channels = sorted({c for c in channel_indices if 0 <= c < size_c})

    cache_file = _cache_file_for(
        path,
        t_index=t,
        z_index=z,
        channel_indices=selected_channels,
    )
    if os.path.exists(cache_file):
        return cache_file

    # Materialize one plane at a time; avoids loading all channels/frames into memory.
    base_plane: NDArray = img.get_image_data("YX", T=t, Z=z, C=0)
    h, w = base_plane.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for rgb_idx, channel_idx in enumerate(selected_channels[:3]):
        plane: NDArray = img.get_image_data("YX", T=t, Z=z, C=channel_idx)
        rgb[:, :, rgb_idx] = _normalize_to_uint8(plane)

    Image.fromarray(rgb, mode="RGB").save(cache_file, format="PNG")
    return cache_file


def fix_java_home_problem():
    # Point JAVA_HOME at jdk4py's bundled OpenJDK 21 (must happen before JVM starts).
    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
    # Prevent scyjava from overriding JAVA_HOME with its own (broken) JVM finder.
    scyjava.config.set_java_constraints(fetch="never")

    # Workaround for jgo 2.1.2 bug on Windows: relativePath in a POM is relative to
    # the POM's *directory*, not the POM file path itself. Without this, jgo tries
    # to parse a directory as XML and raises PermissionError.
    from pathlib import Path
    import jgo.maven._core as _jgo_core
    def _fixed_pom_parent(self, pom):
        from jgo.maven._pom import POM as _POM
        if pom.element("parent") is None:
            return None
        g = pom.value("parent/groupId")
        a = pom.value("parent/artifactId")
        v = pom.value("parent/version")
        assert g and a and v
        relativePath = pom.value("parent/relativePath")
        if (
            isinstance(pom.source, Path)
            and relativePath
            and (parent_path := pom.source.parent / relativePath).exists()
            and parent_path.is_file()
        ):
            parent_pom = _POM(parent_path)
            if g == parent_pom.groupId and a == parent_pom.artifactId and v == parent_pom.version:
                return parent_pom
        pom_artifact = self.project(g, a).at_version(v).artifact(packaging="pom")
        return _POM(pom_artifact.resolve())
    _jgo_core.MavenContext.pom_parent = _fixed_pom_parent




def _configure_bioformats_safe_io(input_path: str) -> None:
    """Best-effort: prevent Bio-Formats from touching the source folder.

    Bio-Formats may write .bfmemo cache files next to inputs (e.g., .ims).
    We try to disable memoization or at least redirect it to a temp dir.
    """
    # Only apply for file types typically handled by Bio-Formats where sidecar writes are common
    if not input_path.lower().endswith((".ims", ".czi", ".lif", ".nd2", ".oib", ".oif")):
        return

    tmp_dir = os.environ.get("BIOFORMATS_MEMO_DIR", None) or tempfile.gettempdir()

    # Set a variety of known env vars / system properties consulted by wrappers
    os.environ.setdefault("BIOFORMATS_DISABLE_MEMOIZATION", "1")
    os.environ.setdefault("OME_BIOFORMATS_MEMOIZER_DISABLED", "1")
    os.environ.setdefault("LOCI_FORMATS_MEMOIZER_DISABLED", "1")
    os.environ.setdefault("BIOFORMATS_MEMO_DIR", tmp_dir)
    os.environ.setdefault("OME_BIOFORMATS_MEMOIZER_DIR", tmp_dir)
    os.environ.setdefault("LOCI_FORMATS_MEMOIZER_DIR", tmp_dir)

    # Reduce Java-side log verbosity (SLF4J/Logback/SciJava) before JVM starts
    try:
        import scyjava  # type: ignore
        # Prepare a minimal Logback configuration to clamp logs to WARN.
        logback_path = os.path.join(tmp_dir, "bioformats-logback.xml")
        if not os.path.exists(logback_path):
            try:
                with open(logback_path, "w", encoding="utf-8") as fh:
                    fh.write(
                        """
<configuration>
    <contextListener class="ch.qos.logback.classic.jul.LevelChangePropagator">
        <resetJUL>true</resetJUL>
    </contextListener>

    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger - %msg%n</pattern>
        </encoder>
    </appender>

    <logger name="loci" level="WARN"/>
    <logger name="ome" level="WARN"/>
    <logger name="org.scijava" level="WARN"/>
    <logger name="org.janelia" level="WARN"/>
    <logger name="net.imagej" level="WARN"/>

    <root level="WARN">
        <appender-ref ref="STDOUT"/>
    </root>
</configuration>
""".strip()
                    )
            except Exception:
                # If writing fails, continue with system properties below
                pass

        # Force Logback to use our configuration if available
        if os.path.exists(logback_path):
            scyjava.config.add_option(f"-Dlogback.configurationFile={logback_path}")

        # SLF4J simple logger (harmless if not the active binding)
        scyjava.config.add_option("-Dorg.slf4j.simpleLogger.defaultLogLevel=warn")
        scyjava.config.add_option("-Dorg.slf4j.simpleLogger.showDateTime=false")
        scyjava.config.add_option("-Dorg.slf4j.simpleLogger.showThreadName=false")
        # SciJava logger level
        scyjava.config.add_option("-Dscijava.log.level=WARN")
        # Fallback envs in case properties aren’t picked up by the active binding
        os.environ.setdefault("SCIJAVA_LOG_LEVEL", "WARN")
    except Exception:
        # If scyjava isn't available yet, the properties below may still be picked up via env when JVM starts
        os.environ.setdefault("org.slf4j.simpleLogger.defaultLogLevel", "warn")
        os.environ.setdefault("scijava.log.level", "WARN")


def _build_bioformats_error_message(path: str, original_error: Exception) -> str:
    """Create a detailed, actionable Bio-Formats initialization error message."""
    chain_messages = []
    seen = set()
    current: Optional[BaseException] = original_error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain_messages.append(str(current))
        next_exc = current.__cause__ if current.__cause__ is not None else current.__context__
        current = next_exc

    error_text = "\n".join(chain_messages)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    is_maven_parent_path_failure = (
        "formats-gpl" in error_text and "Permission denied" in error_text and "..\\.." in error_text
    )

    lines = [
        f"\n{'='*70}",
        f"ERROR: Failed to load {os.path.basename(path)}",
        f"{'='*70}\n",
        "This file format requires Bio-Formats (Java), but Bio-Formats failed to initialize.",
        f"Python environment: {python_version}",
        f"Original error: {original_error}",
        "",
    ]

    if is_maven_parent_path_failure:
        lines.extend([
            "Detected issue: jgo/scyjava Maven dependency resolution failed on Windows",
            "while resolving Bio-Formats parent POMs (path ending in '.pom\\..\\..').",
            "This is not a missing Java executable issue.",
            "",
            "Recommended fixes (in order):",
            "1. Use Python 3.11 for UV env creation (preferred for Bio-Formats on Windows):",
            "   environment: uv@3.11:convert-to-tif",
            "   or set UV_DEFAULT_PYTHON=3.11 before creating the UV env.",
            "2. If problem persists, use Conda environment:",
            "   conda env create -f conda_envs/convert_to_tif.yml",
            "   and use 'environment: convert_to_tif' in your YAML config.",
            "",
        ])
    else:
        lines.extend([
            "Recommended fixes (in order):",
            "1. Recreate this UV env with Python 3.11 (best Bio-Formats compatibility on Windows):",
            "   environment: uv@3.11:convert-to-tif",
            "   or set UV_DEFAULT_PYTHON=3.11 before creating the UV env.",
            "",
            "2. If problem persists, use Conda environment instead of UV:",
            "",
            "   Create Conda environment from: conda_envs/convert_to_tif.yml",
            "   conda env create -f conda_envs/convert_to_tif.yml",
            "",
            "   Run your pipeline with the Conda environment:",
            "   run_pipeline.exe pipeline_configs/your_config.yaml",
            "   (use 'environment: convert_to_tif' in your YAML config)",
            "",
            "NOTE: Most formats (ND2, LIF, CZI, DV, TIFF) work without Bio-Formats.",
            "      Only exotic formats require the Conda environment.",
            "",
        ])

    lines.append(f"{'='*70}\n")
    return "\n".join(lines)


def load_tczyx_image(path: str) -> BioImage:
    """
    Load an image as a BioImage object, ensuring the data is always 5D (TCZYX).
    The file format is determined by the file extension. This function standardizes
    all images to TCZYX order for safe downstream processing.


    example use:
    from bioio import BioImage

    # Get a BioImage object
    img = BioImage("my_file.tiff")  # selects the first scene found
    img.data  # returns 5D TCZYX numpy array
    img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
    img.dims  # returns a Dimensions object
    img.dims.order  # returns string "TCZYX"
    img.dims.X  # returns size of X dimension
    img.shape  # returns tuple of dimension sizes in TCZYX order
    img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

    """
    # Load the image using the appropriate reader based on the file extension
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    lower_path = path.lower()

    if lower_path.endswith((".tif", ".tiff", "ome.tif")):
        is_ome_tiff = False
        
        # Check of this is an OME-TIFF FILE
        try:
            with tifffile.TiffFile(path) as tif:
                is_ome_tiff = bool(getattr(tif, "is_ome", False))
        except Exception:
            # If detection fails, continue with generic reader fallback below.
            pass

        # Only try the OME reader when the TIFF is actually OME-TIFF.
        # This avoids noisy "Failed to parse XML" errors for ImageJ-style TIFFs.
        if is_ome_tiff:
            try:
                import bioio_ome_tiff
                img = BioImage(path, reader=bioio_ome_tiff.Reader)
                return img
            except Exception:
                pass

        # Prefer tifffile reader for non-OME TIFFs (e.g., ImageJ save_mask output).
        try:
            import bioio_tifffile
            img = BioImage(path, reader=bioio_tifffile.Reader)
            return img
        except Exception:
            pass

        # Final generic fallback
        try:
            img = BioImage(path)
            return img
        except Exception:
            pass
        
    elif lower_path.endswith(".nd2"):
        import bioio_nd2
        img = BioImage(path, reader=bioio_nd2.Reader)
        return img
    elif lower_path.endswith(".lif"):
        import bioio_lif
        img = BioImage(path, reader=bioio_lif.Reader)
        return img
    elif lower_path.endswith(".czi"):
        import bioio_czi
        img = BioImage(path, reader=bioio_czi.Reader)
        return img
    elif lower_path.endswith(".dv"):
        import bioio_dv
        img = BioImage(path, reader=bioio_dv.Reader)
        return img
    else:
        import jdk4py
        import scyjava.config
        # Apply Java fixes at import time, before any code can accidentally trigger JVM startup.
        # TODO make this only run once!
        fix_java_home_problem()

        # Unknown format - try Bio-Formats as last resort
        _configure_bioformats_safe_io(path)
        try:
            import bioio_bioformats  # type: ignore
            img = BioImage(path, reader=bioio_bioformats.Reader)
            return img
        except Exception as e:
            raise RuntimeError(_build_bioformats_error_message(path, e)) from e
    raise ValueError(f"Unsupported file format for: {path}")
