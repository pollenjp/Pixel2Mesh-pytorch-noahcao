#!/usr/bin/env bash
# shellcheck shell=bash
set -eux

function download_from_pytorch_wheel() {
    # torch, torchvision
    local package_name="${1}"
    local package_version="${2}"
    local python_version_xy="${3:-3.7}"
    local cuda_version_xy="${4:-9.0}"
    local os_type="${5:-linux}"
    local arch="${6:-x86_64}"

    # version "x.y" to "xy"
    python_version_xy="$(echo "${python_version_xy}" | tr -d .)"
    cuda_version_xy="$(echo "${cuda_version_xy}" | tr -d .)"

    # local pytorch_wheel_name="${package_name}-${package_version}-cp${python_version_xy}-cp${python_version_xy}m-${os_type}_${arch}.whl"
    local pytorch_wheel_name="${package_name}-${package_version}%2Bcu${cuda_version_xy}-cp${python_version_xy}-cp${python_version_xy}-${os_type}_${arch}.whl"
    local whl_path="download.pytorch.org/whl/cu${cuda_version_xy}/${pytorch_wheel_name}"
    local pytorch_wheel_url="https://${whl_path}"
    mkdir -p "${whl_path%/*}"
    if [ ! -f "${whl_path}" ]; then
        wget -O "${whl_path}" "${pytorch_wheel_url}"
    fi
}

function download_pytorch3d_wheel() {
    # torch, torchvision
    local package_version="${1}"
    local python_version_xy="${2}"
    local pytorch_version_xyz="${3}"
    local cuda_version_xy="${4}"
    local os_type="${5:-linux}"
    local arch="${6:-x86_64}"

    # version "x.y" to "xy"
    python_version_xy="$(echo "${python_version_xy}" | tr -d .)"
    pytorch_version_xyz="$(echo "${pytorch_version_xyz}" | tr -d .)"
    cuda_version_xy="$(echo "${cuda_version_xy}" | tr -d .)"

    # https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/pytorch3d-0.7.1-cp38-cp38-linux_x86_64.whl
    # https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1120/pytorch3d-0.7.1-cp39-cp39-linux_x86_64.whl
    # https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu116_pyt1121/pytorch3d-0.7.1-cp310-cp310-linux_x86_64.whl
    local wheel_file_name="pytorch3d-${package_version}-cp${python_version_xy}-cp${python_version_xy}-${os_type}_${arch}.whl"
    local whl_path="dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${python_version_xy}_cu${cuda_version_xy}_pyt${pytorch_version_xyz}/${wheel_file_name}"
    local pytorch_wheel_url="https://${whl_path}"
    mkdir -p "${whl_path%/*}"
    if [ ! -f "${whl_path}" ]; then
        wget -O "${whl_path}" "${pytorch_wheel_url}"
    fi
}

(
    base_dir=".wheel"
    mkdir -p "${base_dir}"
    cd "${base_dir}"

    python_version_xy="3.10"
    torch_version_xyz="1.12.1"
    torchvision_version_xyz="0.13.1"
    cuda_version_xy="11.3"

    download_from_pytorch_wheel torch "${torch_version_xyz}" "${python_version_xy}" "${cuda_version_xy}"
    download_from_pytorch_wheel torchvision "${torchvision_version_xyz}" "${python_version_xy}" "${cuda_version_xy}"

    pytorch3d_version_xyz="0.7.2"
    download_pytorch3d_wheel "${pytorch3d_version_xyz}" "${python_version_xy}" "${torch_version_xyz}" "${cuda_version_xy}"
)
