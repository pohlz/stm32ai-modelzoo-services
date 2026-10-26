#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)][string]$ModelFolder,
    [Parameter(Mandatory=$true, Position=1)][string]$MemoryPool,
    [string]$StEdgeAi = 'C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe'
)
$ErrorActionPreference = 'Stop'
try {
    # Accept either a path or an experiment name, as with stedgeai_analyze.
    $run = $ModelFolder
    if (-not (Test-Path -LiteralPath $run -PathType Container)) {
        $run = Join-Path $PSScriptRoot "tf\src\experiments_outputs\$ModelFolder"
    }
    $run = (Resolve-Path -LiteralPath $run).Path
    $pool = (Resolve-Path -LiteralPath $MemoryPool).Path
    if (-not (Test-Path -LiteralPath $pool -PathType Leaf)) { throw 'MemoryPool must be a file.' }
    $exe = (Get-Command $StEdgeAi -ErrorAction Stop).Source
    $sdk = Split-Path (Split-Path (Split-Path $exe -Parent) -Parent) -Parent
    $runtime = Join-Path $sdk 'Middlewares\ST\AI'
    if (-not (Test-Path -LiteralPath $runtime -PathType Container)) {
        throw "Matching STM32 AI middleware was not found: $runtime"
    }
    $models = @(Get-ChildItem -LiteralPath $run -Recurse -File | Where-Object {
        $_.Name -in @('quantize_model.tflite', 'quantized_model.tflite') -and
        $_.FullName -notmatch '[\\/]NeuralART_compilation[^\\/]*[\\/]'
    })
    if ($models.Count -ne 1) {
        throw "Expected exactly one quantize_model.tflite or quantized_model.tflite under $run; found $($models.Count). Pass a more specific folder."
    }
    $out = Join-Path $run ('NeuralART_compilation_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
    $poolCopy = Join-Path $out 'memory_pool.mpool'
    Copy-Item -LiteralPath $pool -Destination $poolCopy
    $profile = Join-Path $out 'neuralart.json'
    @{ Profiles = @{ manual = @{
        memory_pool = $poolCopy.Replace('\','/')
        options = '-O3 --all-buffers-info --cache-maintenance --Oauto-sched --native-float --enable-virtual-mem-pools --Omax-ca-pipe 4 --Ocache-opt --Os'
    } } } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $profile -Encoding ASCII
    $generated = Join-Path $out 'generated'
    $workspace = Join-Path $out 'workspace'
    $cliArgs = @('generate', '--model', $models[0].FullName, '--target', 'stm32n6',
        '--name', 'network', '--c-api', 'st-ai', '--st-neural-art', "manual@$profile",
        '--optimization', 'balanced', '--output', $generated, '--workspace', $workspace)
    @{ executable=$exe; arguments=$cliArgs; model=$models[0].FullName; memory_pool=$pool } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out 'command.json') -Encoding UTF8
    Write-Host "Compiling: $($models[0].FullName)"
    Write-Host "Output:    $out"
    # Windows PowerShell otherwise treats native stderr progress messages as errors.
    $ErrorActionPreference = 'Continue'
    & $exe @cliArgs 2>&1 | Tee-Object -FilePath (Join-Path $out 'stedgeai_compile.log')
    $status = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($status -ne 0) {
        $logText = Get-Content -LiteralPath (Join-Path $out 'stedgeai_compile.log') -Raw
        if ($logText -match 'total bytes left unallocated\s*=\s*(\d+)') {
            throw "Neural-ART memory allocation failed: $($Matches[1]) bytes left unallocated with memory pool '$pool'. Supply a pool matching sufficient available board memory, or reduce the model's memory requirements. See $out\stedgeai_compile.log"
        }
        throw "STEdgeAI failed with exit code $status. See $out\stedgeai_compile.log"
    }
    foreach ($required in @('network.c', 'network.h', 'stai_network.c', 'stai_network.h')) {
        if (-not (Test-Path -LiteralPath (Join-Path $generated $required))) { throw "Missing generated file: $required" }
    }
    # FORMAT_RAW pool initializers are already binary bytes, not text hex.
    # Keep originals and provide .bin copies for STM32CubeProgrammer.
    Get-ChildItem -LiteralPath $generated -Recurse -File -Filter '*.raw' | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination ([IO.Path]::ChangeExtension($_.FullName, '.bin'))
    }
    $weights = @(Get-ChildItem -LiteralPath $generated -Recurse -File -Filter '*.bin')
    if ($weights.Count -eq 0) { throw 'Compiler produced no binary weights. Inspect the log and memory-pool configuration.' }
    # Generate metadata for every initializer, including writable SRAM pools.
    & (Join-Path $PSScriptRoot 'stedgeai_weights_header.ps1') -Generated $generated -MemoryPool $poolCopy
    # Include the exact SDK middleware, headers, libraries, and license for manual integration.
    $middleware = Join-Path $out 'Middlewares\ST'
    New-Item -ItemType Directory -Path $middleware | Out-Null
    Copy-Item -LiteralPath $runtime -Destination $middleware -Recurse
    $weights | ForEach-Object { $_.FullName.Substring($out.Length + 1) } |
        Set-Content -LiteralPath (Join-Path $out 'weights_manifest.txt')
    @'
Manual STM32Cube integration (STM32N6 / Neural-ART)

generated/ contains the generated model sources, headers, reports and weights.
workspace/ retains all intermediate compiler output, including per-pool assets.
weights_manifest.txt lists the binary files produced by the compiler.
memory_pool.mpool and neuralart.json record the memory layout and compiler options.
Middlewares/ST/AI contains the matching installed SDK runtime and license files.

1. Copy generated C/H model files into your project's model source/include folders.
   Add the model include directory and compile the generated C sources.
   Copy generated/audio_model_weights.h to your application's audio include
   folder and compile generated/audio_model_weights.c in your project. Its address, full binary size and CRC32
   match this compilation's initializer binary, whether in flash or SRAM.
   For SRAM placement, store the initializer in persistent storage and copy it
   to the generated address after enabling SRAM and before checking the CRC
   or initializing/running the model. SRAM contents are lost across power-off.
   Program SRAM initializer binaries in ascending destination order at flash
   addresses 0x70400000 and 0x70480000 respectively (512 KiB slots). The generated
   SOURCE_ADDRESS defaults can be overridden in compiler flags. Then call
   AudioModelWeights_Init() and check its return value. Flash-only models need
   no source configuration. Init copies SRAM sections and verifies all CRCs.
   AudioModelWeights_CopyAll() also takes no arguments and uses sources[].
   The generated array has one {pointer, size} per section; {NULL, 0} for flash
   already programmed in place. SRAM pointers identify persistent storage.
   The helper copies SRAM sections, cleans CPU D-cache and verifies every CRC.
   The same Init/CopyAll/Verify functions handle one or multiple sections.
   For multiple sections, replace legacy single-range CRC checks with
   AudioModelWeights_Verify(). Single-range macros exist only for one section.
   The source address is your flash STORAGE location, not the SRAM destination.
   If weights and activations share the initializer range, verify before inference.
2. Integrate the matching ST AI / Neural-ART runtime from Middlewares/ST/AI.
   Use its STM32N6 toolchain libraries and required runtime sources/includes;
   do not compile every middleware source or mix runtime versions.
3. Reserve the activation/weight regions described by memory_pool.mpool in your
   linker script. Configure clocks, NPU, memory controllers, MPU and caches in
   the board application, using an STM32N6 Neural-ART project as the baseline.
4. Program the per-pool binary weights into the addresses specified by the
   generated network.c / compiler reports and memory pool. A .bin file does
   not encode its flash address. Check each pool's address and board layout.
   Preserve any additional compiler-generated initializer/epoch assets.
5. Initialize and invoke the model through generated stai_network.h, following
   the matching SDK example. Use the generated tensor types/shapes/quantization.

This package contains model artifacts and SDK middleware. Board startup, linker
layout, external-memory loader and application inference code remain specific
to your STM32Cube project. Generation does not build or flash firmware.
'@ | Set-Content -LiteralPath (Join-Path $out 'INTEGRATION.txt') -Encoding ASCII
    Write-Host "Compilation and packaging completed: $out"
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
