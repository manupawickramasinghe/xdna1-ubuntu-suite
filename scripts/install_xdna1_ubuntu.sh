#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# install_xdna1_ubuntu.sh — build and install XRT + the amdxdna XRT plugin for
# 1st-gen Ryzen AI NPUs (Phoenix / Hawk Point, PCI 1022:1502) on Ubuntu 24.04+
# and derivatives (Pop!_OS, Linux Mint, ...).
#
# Stages (default: all, in order):
#   check     hardware / kernel / IOMMU / permissions preflight (no changes)
#   deps      [sudo] apt build deps, memlock limits, render group
#   build     [user] clone amd/xdna-driver, build XRT (-npu) and the plugin
#   install   [sudo] install the generated .deb packages
#   validate  [user] xrt-smi examine / validate
#
#   scripts/install_xdna1_ubuntu.sh                 # everything
#   scripts/install_xdna1_ubuntu.sh check build     # selected stages
#   XDNA_WORKDIR=/data/xdna scripts/install_xdna1_ubuntu.sh
#
# Kernel driver: Linux >= 6.14 ships amdxdna in-tree. On such kernels the plugin
# is built with -nokmod so DKMS doesn't build a second, out-of-tree amdxdna.ko.
# Force the DKMS driver with XDNA_KMOD=1.

set -euo pipefail

WORKDIR="${XDNA_WORKDIR:-$HOME/.cache/xdna1-build}"
SRC="$WORKDIR/xdna-driver"
XDNA_REF="${XDNA_REF:-main}"        # branch/tag/commit of amd/xdna-driver
JOBS="${JOBS:-$(nproc)}"

c_bold=$'\e[1m'; c_red=$'\e[31m'; c_grn=$'\e[32m'; c_yel=$'\e[33m'; c_off=$'\e[0m'
step() { echo; echo "${c_bold}=== $* ===${c_off}"; }
ok()   { echo "  ${c_grn}✓${c_off} $*"; }
warn() { echo "  ${c_yel}!${c_off} $*"; }
die()  { echo "  ${c_red}✗ $*${c_off}" >&2; exit 1; }

kernel_ge() { # kernel_ge 6 14
	local maj min; IFS=. read -r maj min _ <<<"$(uname -r)"
	(( maj > $1 || (maj == $1 && min >= $2) ))
}

os_id()      { . /etc/os-release; echo "${ID:-}"; }
os_id_like() { . /etc/os-release; echo "${ID_LIKE:-}"; }

# ---------------------------------------------------------------------------
stage_check() {
	step "[check] Hardware, kernel and permissions"
	local fail=0

	local pci; pci="$(lspci -nn)"
	if grep -qi "1022:1502" <<<"$pci"; then ok "XDNA1 NPU (1022:1502) present: $(grep -i 1022:1502 <<<"$pci" | cut -d' ' -f1)"
	elif grep -qi "1022:17f0" <<<"$pci"; then warn "XDNA2 NPU (1022:17f0) found; this script targets XDNA1 but XRT works the same"
	else die "No AMD NPU on the PCI bus. Enable 'IPU' / 'NPU' in the BIOS (AMD CBS → NBIO)."; fi

	if kernel_ge 6 14; then ok "kernel $(uname -r) has in-tree amdxdna"
	elif kernel_ge 6 10; then warn "kernel $(uname -r): no in-tree driver, the DKMS module from the plugin package will be used"
	else die "kernel $(uname -r) is too old (need >= 6.10; Ubuntu 24.04: sudo apt install linux-generic-hwe-24.04)"; fi

	if [[ -d /sys/module/amdxdna ]]; then ok "amdxdna module loaded"
	else warn "amdxdna not loaded (try: sudo modprobe amdxdna)"; fi

	if [[ -e /sys/class/accel/accel0 ]]; then
		ok "/dev/accel/accel0 present, fw $(cat /sys/class/accel/accel0/device/fw_version 2>/dev/null || echo '?')"
	else warn "/dev/accel/accel0 missing (driver not bound; check: journalctl -k | grep -i amdxdna)"; fail=1; fi

	if compgen -G "/sys/class/iommu/*" >/dev/null; then ok "IOMMU active ($(ls /sys/class/iommu | tr '\n' ' '))"
	else warn "no IOMMU in /sys/class/iommu; enable IOMMU in BIOS (SVA is required)"; fail=1; fi
	if grep -q "SVA bind device failed" <<<"$(journalctl -k -b --no-pager 2>/dev/null)"; then
		warn "kernel log has 'SVA bind device failed': IOMMU SVA isn't working"; fail=1
	fi

	if [[ " $(id -nG) " == *" render "* ]]; then ok "user in 'render' group"
	else warn "user not in 'render' group (deps stage fixes this; log out and back in after)"; fi

	local ml; ml="$(ulimit -l)"
	if [[ "$ml" == unlimited ]]; then ok "memlock unlimited"
	else warn "memlock is ${ml} KiB; large BOs need 'unlimited' (deps stage fixes this)"; fi

	. /etc/os-release
	ok "distro: ${PRETTY_NAME:-$ID} (ID=$ID${ID_LIKE:+, ID_LIKE=$ID_LIKE})"
	if [[ -x /opt/xilinx/xrt/bin/xrt-smi ]]; then ok "XRT already installed at /opt/xilinx/xrt"; fi
	return $fail
}

# ---------------------------------------------------------------------------
# XRT/xdna-driver build scripts only recognise ID=ubuntu|debian|linuxmint.
# On Ubuntu derivatives (Pop!_OS: ID=pop, ID_LIKE=ubuntu) point them at a
# copy of os-release with ID=ubuntu. Only touches files inside $SRC.
flavor_shim_path() { echo "$WORKDIR/os-release.ubuntu"; }

need_flavor_shim() {
	local id; id="$(os_id)"
	[[ "$id" != ubuntu && "$id" != debian && "$id" != linuxmint ]] && [[ " $(os_id_like) " == *" ubuntu "* ]]
}

write_flavor_shim() {
	mkdir -p "$WORKDIR"
	sed -E 's/^ID=.*/ID=ubuntu/' /etc/os-release > "$(flavor_shim_path)"
}

apply_flavor_shim() {
	need_flavor_shim || return 0
	write_flavor_shim
	local shim; shim="$(flavor_shim_path)"
	local files=(
		"$SRC/build/build.sh"
		"$SRC/CMake/pkg.cmake"
		"$SRC/xrt/build/build.sh"
		"$SRC/xrt/src/CMake/nativeLnx.cmake"
		"$SRC/xrt/src/CMake/cpackLin.cmake"
		"$SRC/xrt/src/runtime_src/tools/scripts/xrtdeps.sh"
	)
	for f in "${files[@]}"; do
		[[ -f "$f" ]] && sed -i "s|/etc/os-release|$shim|g" "$f"
	done
	ok "applied $(os_id) → ubuntu flavor shim to XRT/xdna-driver build scripts"
}

# xclbinutil's gtest unit test links against whichever GoogleTest CMake finds
# first; a hand-built gtest in /usr/local next to the distro libgtest-dev gives
# header/library mismatches (undefined MakeAndRegisterTestInfo). The tests
# aren't needed for an install, so skip that lookup in our checkout.
apply_build_fixes() {
	local f="$SRC/xrt/src/runtime_src/tools/xclbinutil/CMakeLists.txt"
	if [[ -f "$f" ]] && grep -q '^find_package(GTest)' "$f"; then
		sed -i 's/^find_package(GTest)/set(GTEST_FOUND OFF) # xdna1-ubuntu-suite: unit tests disabled/' "$f"
		ok "disabled xclbinutil gtest unit test build"
	fi
}

# ---------------------------------------------------------------------------
clone_sources() {
	if ! command -v git-lfs >/dev/null; then die "git-lfs missing; run the 'deps' stage first"; fi
	git lfs install --skip-repo >/dev/null
	mkdir -p "$WORKDIR"
	if [[ ! -d "$SRC/.git" ]]; then
		git clone https://github.com/amd/xdna-driver.git "$SRC"
	fi
	git -C "$SRC" fetch --tags origin
	# Drop any previous flavor shim edits before switching revisions.
	git -C "$SRC" submodule foreach --recursive -q 'git checkout -q -- . 2>/dev/null || true'
	git -C "$SRC" checkout -q -- .
	git -C "$SRC" checkout -q "$XDNA_REF"
	git -C "$SRC" pull -q --ff-only 2>/dev/null || true
	git -C "$SRC" submodule update --init --recursive
	ok "amd/xdna-driver @ $(git -C "$SRC" rev-parse --short HEAD), xrt @ $(git -C "$SRC/xrt" rev-parse --short HEAD)"
}

stage_deps() {
	step "[deps] Build dependencies, memlock, groups (sudo)"
	sudo apt-get update
	sudo apt-get install -y git git-lfs jq build-essential cmake pkg-config dkms \
		"linux-headers-$(uname -r)" python3-dev pybind11-dev
	git lfs install --skip-repo >/dev/null

	# XRT's own dependency script needs the sources; clone first (as the user).
	clone_sources
	apply_flavor_shim
	sudo "$SRC/xrt/src/runtime_src/tools/scripts/xrtdeps.sh"
	ok "XRT build dependencies installed"

	sudo install -d /etc/security/limits.d
	sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null <<-'EOF'
	# Locked memory for AMD XDNA NPU buffer objects (xdna1-ubuntu-suite)
	*	soft	memlock	unlimited
	*	hard	memlock	unlimited
	EOF
	ok "memlock unlimited via /etc/security/limits.d/99-amdxdna.conf"

	sudo usermod -aG render "$USER"
	ok "added $USER to 'render'"
	warn "log out and back in for the group + memlock changes to apply"
}

stage_build() {
	step "[build] XRT (-npu) and amdxdna plugin (as $USER)"
	[[ $EUID -ne 0 ]] || die "run the build stage as a normal user, not root"
	clone_sources
	apply_flavor_shim
	apply_build_fixes

	( cd "$SRC/xrt/build" && ./build.sh -npu -opt -noctest -j "$JOBS" )
	ok "XRT packages: $(ls "$SRC"/xrt/build/Release/xrt_*.deb 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"

	local kmod_flag=()
	if kernel_ge 6 14 && [[ "${XDNA_KMOD:-0}" != 1 ]]; then kmod_flag=(-nokmod); fi
	# The plugin build compiles XRT from the submodule itself; it doesn't need /opt/xilinx/xrt.
	( cd "$SRC/build" && ./build.sh -release "${kmod_flag[@]}" -j "$JOBS" )
	ok "plugin packages: $(ls "$SRC"/build/Release/xrt_plugin*.deb 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
}

stage_install() {
	step "[install] Install packages (sudo)"
	local xrt_debs=("$SRC"/xrt/build/Release/xrt_*.deb)
	[[ -e "${xrt_debs[0]}" ]] || die "no XRT .deb found; run the build stage"
	sudo apt-get install -y --reinstall "${xrt_debs[@]}"
	ok "installed XRT"

	local plugin_debs=("$SRC"/build/Release/xrt_plugin*.deb)
	[[ -e "${plugin_debs[0]}" ]] || die "no plugin .deb found; run the build stage"
	sudo apt-get install -y --reinstall "${plugin_debs[@]}"
	ok "installed amdxdna XRT plugin"
}

stage_validate() {
	step "[validate] xrt-smi"
	[[ -f /opt/xilinx/xrt/setup.sh ]] || die "XRT not installed"
	# shellcheck disable=SC1091
	set +u; source /opt/xilinx/xrt/setup.sh >/dev/null; set -u
	xrt-smi examine
	if [[ "$(ulimit -l)" != unlimited ]]; then
		warn "memlock isn't unlimited in this shell yet; log out/in if validate fails with 'Lock limit exceeded'"
	fi
	xrt-smi validate || die "xrt-smi validate failed"
	ok "NPU validated. Try: build/xdna_daemon --benchmark-loop <xclbin> and watch btop"
}

# ---------------------------------------------------------------------------
stages=("$@")
[[ ${#stages[@]} -gt 0 ]] || stages=(check deps build install validate)
for s in "${stages[@]}"; do
	case "$s" in
		check)    stage_check || warn "preflight reported problems (continuing)";;
		deps)     stage_deps;;
		build)    stage_build;;
		install)  stage_install;;
		validate) stage_validate;;
		-h|--help) sed -n '2,24p' "$0"; exit 0;;
		*) die "unknown stage '$s' (check|deps|build|install|validate)";;
	esac
done
