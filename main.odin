// The code here follows the tutorial from:
// https://vulkan-tutorial.com

package main

import "core:log"
import "core:os"
import "rendering"
import "vendor:glfw"
import vk "vendor:vulkan"

WIDTH :: 1024
HEIGHT :: 768
TITLE :: "Hello Vulkan!"

when ODIN_OS == .Darwin {
	// NOTE: just a bogus import of the system library,
	// needed so we can add a linker flag to point to /usr/local/lib (where vulkan is installed by default)
	// when trying to load vulkan.
	// Credit goes to : https://gist.github.com/laytan/ba57af3e5a59ab5cb2fca9e25bcfe262
	@(require, extra_linker_flags = "-rpath /usr/local/lib")
	foreign import __ "system:System.framework"
}

main :: proc() {
	os.exit(int(run()))
}

run :: proc() -> vk.Result {
	context.logger = log.create_console_logger()
	if !bool(glfw.Init()) {
		return .ERROR_INITIALIZATION_FAILED
	}
	defer glfw.Terminate()
	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	window := glfw.CreateWindow(WIDTH, HEIGHT, TITLE, nil, nil)
	if window == nil {
		return .ERROR_INITIALIZATION_FAILED
	}
	defer glfw.DestroyWindow(window)
	renderer: rendering.Renderer
	rendering.init(&renderer, window)
	defer rendering.destroy(&renderer)
	for !glfw.WindowShouldClose(window) {
		glfw.PollEvents()
		rendering.update(&renderer)
		rendering.render(&renderer) or_continue
	}
	return .SUCCESS
}
