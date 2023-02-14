ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CMD_DIR := ${ROOT}/.local
BLENDER_DIR := ${CMD_DIR}/blender

# <https://download.blender.org/release/>
BLENDER_VERSION_MAJOR_MINOR := 2.93
BLENDER_CMD := ${BLENDER_DIR}/blender

OUTPUT_DIR := output
DEBUG_BLEND_FILE := ${OUTPUT_DIR}/debug.blend

RUN_PY_FILE := main.py

.PHONY: setup
setup:
	./setup.sh

.PHONY: resetup
resetup:
	rm -rf "${BLENDER_DIR}"
	${MAKE} setup

.PHONY: run
run:
	${BLENDER_CMD} \
		--background \
		--python \
		"${RUN_PY_FILE}" \
			-- \
			obj_filepath="sample.obj" \
			output_dirpath="out" \
			output_name="out"

.PHONY: clean
clean:
	rm -rf "${CMD_DIR}"
