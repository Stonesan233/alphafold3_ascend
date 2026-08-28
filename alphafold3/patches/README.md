# Offline build patches

## dssp-v4.4.7-mkdssp-option.patch

Applies to PDB-REDO/dssp @ `57560472b4260dc41f457706bc45fc6ef0bc0f10` (v4.4.7),
the exact revision fetched by the root `CMakeLists.txt`.

Problem: dssp builds the `mkdssp` executable and links `libmcfp::libmcfp`;
newer libmcfp no longer provides that target, breaking offline builds.

The patch adds a `DSSP_BUILD_MKDSSP` option (default `ON`, preserving upstream
behaviour). AlphaFold3 only needs the `dssp::dssp` library, so configure with:

```bash
git apply patches/dssp-v4.4.7-mkdssp-option.patch   # inside the dssp source
# or, when preparing $ALPHAFOLD3_DEPS_DIR/dssp
cmake -DDSSP_BUILD_MKDSSP=OFF ...
```

Note: this patch must be applied to the **dssp third-party source**, not to
this repository. When preparing `$ALPHAFOLD3_DEPS_DIR`, the `dssp/` directory
must be the patched copy.
