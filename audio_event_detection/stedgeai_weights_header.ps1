#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Generated,
    [Parameter(Mandatory=$true)][string]$MemoryPool
)
$ErrorActionPreference = 'Stop'
$description = Get-Content -LiteralPath $MemoryPool -Raw | ConvertFrom-Json
if (-not ('AudioWeightsCrc32' -as [type])) {
    Add-Type -TypeDefinition @'
public static class AudioWeightsCrc32 {
    public static uint Compute(byte[] data) {
        uint crc = 0xFFFFFFFFU;
        foreach (byte value in data) {
            crc ^= value;
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ ((crc & 1U) != 0 ? 0xEDB88320U : 0U);
        }
        return crc ^ 0xFFFFFFFFU;
    }
}
'@
}
$sections = @()
foreach ($pool in $description.memory.mempools) {
    $name = 'network_{0}.{1}.bin' -f $description.memory.mem_file_prefix, $pool.fname
    $path = Join-Path $Generated $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    if ($pool.mode -eq 'USEMODE_RELATIVE' -or $pool.offset.magnitude -ne 'BYTES') {
        throw "Initializer $name requires an absolute address in BYTES."
    }
    $value = [string]$pool.offset.value
    $address = if ($value -match '^0x') { [Convert]::ToUInt64($value.Substring(2),16) } else { [Convert]::ToUInt64($value,10) }
    $bytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $path).Path)
    if ($bytes.Length -eq 0 -or $address -eq 0 -or ($address + $bytes.Length) -gt 0x100000000L) { throw "Invalid initializer range: $name" }
    if ($pool.prop.rights -notin @('ACC_READ','ACC_WRITE')) { throw "Unsupported initializer access rights: $name" }
    $sections += [pscustomobject]@{ Name=$name; Address=$address; Size=$bytes.Length; Crc=[AudioWeightsCrc32]::Compute($bytes); Ram=[int]($pool.prop.rights -eq 'ACC_WRITE') }
}
$sections = @($sections | Sort-Object Address)
$bins = @(Get-ChildItem -LiteralPath $Generated -File -Filter 'network_*.bin')
if ($sections.Count -eq 0 -or $sections.Count -ne $bins.Count) { throw 'Every initializer binary must match a memory-pool entry; virtual/unknown pools need explicit address resolution.' }
if (@($sections | Where-Object Ram -eq 1).Count -gt 2 -or @($sections | Where-Object Ram -eq 0).Count -gt 1) { throw 'Supported layouts: at most one flash initializer and up to two SRAM initializers.' }
for ($i=1; $i -lt $sections.Count; $i++) {
    if ($sections[$i].Address -lt ($sections[$i-1].Address + $sections[$i-1].Size)) { throw 'Initializer destination ranges overlap.' }
}
$defines = @("#define AUDIO_MODEL_WEIGHTS_SECTION_COUNT $($sections.Count)U")
$rows = @()
for ($i=0; $i -lt $sections.Count; $i++) {
    $s = $sections[$i]
    $defines += "/* $($s.Name) */"
    $defines += "#define AUDIO_MODEL_WEIGHTS_SECTION_${i}_ADDRESS 0x$('{0:X8}' -f $s.Address)UL"
    $defines += "#define AUDIO_MODEL_WEIGHTS_SECTION_${i}_SIZE $($s.Size)UL"
    $defines += "#define AUDIO_MODEL_WEIGHTS_SECTION_${i}_CRC32 0x$('{0:X8}' -f $s.Crc)UL"
    $defines += "#define AUDIO_MODEL_WEIGHTS_SECTION_${i}_IN_RAM $($s.Ram)"
    $rows += "    { AUDIO_MODEL_WEIGHTS_SECTION_${i}_ADDRESS, AUDIO_MODEL_WEIGHTS_SECTION_${i}_SIZE, AUDIO_MODEL_WEIGHTS_SECTION_${i}_CRC32, AUDIO_MODEL_WEIGHTS_SECTION_${i}_IN_RAM }"
}
if ($sections.Count -eq 1) {
    foreach ($field in @('ADDRESS','SIZE','CRC32','IN_RAM')) { $defines += "#define AUDIO_MODEL_WEIGHTS_$field AUDIO_MODEL_WEIGHTS_SECTION_0_$field" }
}
$sourceDefines = @()
$sourceRows = @()
$ramSourceIndex = 0
for ($i=0; $i -lt $sections.Count; $i++) {
    if ($sections[$i].Ram) {
        $sourceAddress = 0x70400000L + ($ramSourceIndex * 0x80000L)
        if ($sections[$i].Size -gt 0x80000) { throw 'SRAM initializer exceeds its 512 KiB flash storage slot.' }
        foreach ($flashSection in @($sections | Where-Object Ram -eq 0)) {
            if ($sourceAddress -lt ($flashSection.Address + $flashSection.Size) -and $flashSection.Address -lt ($sourceAddress + $sections[$i].Size)) {
                throw 'Default SRAM initializer storage overlaps in-place flash weights. Use a layout with separate flash storage regions.'
            }
        }
        $sourceDefines += "/* $($sections[$i].Name) -> destination SECTION_$i */"
        $sourceDefines += "#ifndef AUDIO_MODEL_WEIGHTS_SECTION_${i}_SOURCE_ADDRESS"
        $sourceDefines += "#define AUDIO_MODEL_WEIGHTS_SECTION_${i}_SOURCE_ADDRESS 0x$('{0:X8}' -f $sourceAddress)UL"
        $sourceDefines += "#endif"
        $sourceRows += "    { (const void *)(uintptr_t)AUDIO_MODEL_WEIGHTS_SECTION_${i}_SOURCE_ADDRESS, AUDIO_MODEL_WEIGHTS_SECTION_${i}_SIZE }"
        $ramSourceIndex++
    } else {
        $sourceRows += "    { NULL, 0U } /* $($sections[$i].Name): programmed in place */"
    }
}
$header = @'
/* Generated model initializer interface. Compile audio_model_weights.c too. */
#ifndef AUDIO_MODEL_WEIGHTS_H
#define AUDIO_MODEL_WEIGHTS_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
@DEFINES@
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    uintptr_t address;
    size_t size;
    uint32_t crc32;
    bool in_ram;
} AudioModelWeightsSection;
typedef struct {
    const void *data;
    size_t size;
} AudioModelWeightsSource;
extern const AudioModelWeightsSection audio_model_weights_sections[AUDIO_MODEL_WEIGHTS_SECTION_COUNT];
/* SRAM sources default to 0x70400000 and 0x70480000 (destination address order).
 * Override SOURCE_ADDRESS macros in audio_model_weights.c or build flags.
 * Call after enabling/mapping memories, before any model/NPU initialization.
 * Returns false if sources are invalid or any destination CRC mismatches. */
bool AudioModelWeights_Init(void);
bool AudioModelWeights_Verify(void);
bool AudioModelWeights_CopyAll(void);
#ifdef __cplusplus
}
#endif
#endif
'@
$source = @'
/* Generated implementation. Keep with matching model/binaries and header. */
#include "audio_model_weights.h"
#include "stm32n6xx.h"
#include <string.h>

/* SRAM initializer flash storage slots: first 0x70400000, second 0x70480000.
 * Program each listed binary in persistent memory, then set its source address.
 * Defaults follow ascending SRAM destination address; each slot is 512 KiB.
 * Flash destinations are programmed in place; they need no source address. */
@SOURCES@
const AudioModelWeightsSection audio_model_weights_sections[AUDIO_MODEL_WEIGHTS_SECTION_COUNT] = {
@ROWS@
};
static const AudioModelWeightsSource sources[AUDIO_MODEL_WEIGHTS_SECTION_COUNT] = {
@SOURCE_ROWS@
};

bool AudioModelWeights_Init(void)
{
    return AudioModelWeights_CopyAll();
}

/* Verify all destination sections before model initialization/inference.
 * Also use this instead of a legacy single-range check for multi-section models. */
bool AudioModelWeights_Verify(void)
{
    for (size_t i = 0; i < AUDIO_MODEL_WEIGHTS_SECTION_COUNT; ++i) {
        const AudioModelWeightsSection *s = &audio_model_weights_sections[i];
        const volatile uint8_t *p = (const volatile uint8_t *)s->address;
        uint32_t crc = 0xFFFFFFFFUL;
        for (size_t j = 0; j < s->size; ++j) {
            crc ^= p[j];
            for (unsigned bit = 0; bit < 8; ++bit)
                crc = (crc >> 1U) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
        }
        if ((crc ^ 0xFFFFFFFFUL) != s->crc32) return false;
    }
    return true;
}

/* Call once after SRAM/flash mapping and permissions are configured, before
 * any NPU access. Uses the generated sources[] array in this file.
 * For flash sections use {NULL,0}: they must already be programmed in place.
 * For RAM sections provide the corresponding binary's flash storage pointer
 * and byte length. Sources must not overlap ANY RAM destination (including
 * cache-line padding). Reserve destination cache lines for the model.
 * Do not call during inference or after NPU cache has accessed these buffers.
 * False may mean CRC failure after copying; never run the model on failure. */
bool AudioModelWeights_CopyAll(void)
{
    const size_t count = AUDIO_MODEL_WEIGHTS_SECTION_COUNT;
    /* Validate all sources before modifying any SRAM destination. */
    for (size_t i = 0; i < count; ++i) {
        const AudioModelWeightsSection *s = &audio_model_weights_sections[i];
        if (!s->in_ram) {
            if (sources[i].data != NULL && (uintptr_t)sources[i].data != s->address) return false;
            continue;
        }
        uintptr_t src = (uintptr_t)sources[i].data;
        if (src == 0 || sources[i].size < s->size || src > UINTPTR_MAX - s->size) return false;
        for (size_t j = 0; j < count; ++j) {
            const AudioModelWeightsSection *d = &audio_model_weights_sections[j];
            if (d->in_ram && src < d->address + d->size && d->address < src + s->size) return false;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const AudioModelWeightsSection *s = &audio_model_weights_sections[i];
        if (!s->in_ram) continue;
        memcpy((void *)s->address, sources[i].data, s->size);
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
        uintptr_t mask = (uintptr_t)__SCB_DCACHE_LINE_SIZE - 1U;
        uintptr_t start = s->address & ~mask;
        uintptr_t end = (s->address + s->size + mask) & ~mask;
        SCB_CleanDCache_by_Addr((void *)start, (int32_t)(end - start));
#endif
    }
    __DSB();
    return AudioModelWeights_Verify();
}
'@
$header.Replace('@DEFINES@',($defines -join "`r`n")) |
    Set-Content -LiteralPath (Join-Path $Generated 'audio_model_weights.h') -Encoding ASCII
$source.Replace('@SOURCES@',($sourceDefines -join "`r`n")).Replace('@SOURCE_ROWS@',($sourceRows -join ",`r`n")).Replace('@ROWS@',($rows -join ",`r`n")) |
    Set-Content -LiteralPath (Join-Path $Generated 'audio_model_weights.c') -Encoding ASCII
