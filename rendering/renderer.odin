package rendering
import "base:runtime"
import "core:log"
import "core:slice"
import "core:strings"
import "vendor:glfw"
import vk "vendor:vulkan"

TITLE :: "Hello Vulkan!"
ENGINE_NAME :: "Soreal Engine"
ENABLE_VALIDATION_LAYERS :: #config(ENABLE_VALIDATION_LAYERS, ODIN_DEBUG)
REQUIRED_EXTENSIONS :: []cstring{vk.KHR_SWAPCHAIN_EXTENSION_NAME}
when ENABLE_VALIDATION_LAYERS {
	LAYERS :: []cstring{"VK_LAYER_KHRONOS_validation"}
} else {
	LAYERS :: []cstring{}
}
SHADER_VERT :: #load("shaders/vert.spv")
SHADER_FRAG :: #load("shaders/frag.spv")
MAX_FRAMES_IN_FLIGHT :: 2

Renderer :: struct {
	instance:                   vk.Instance,
	window:                     glfw.WindowHandle,
	dbg_messenger:              vk.DebugUtilsMessengerEXT,
	surface:                    vk.SurfaceKHR,
	physical_device:            vk.PhysicalDevice,
	device:                     vk.Device,
	graphics_queue:             vk.Queue,
	present_queue:              vk.Queue,
	swapchain:                  vk.SwapchainKHR,
	swapchain_images:           []vk.Image,
	swapchain_views:            []vk.ImageView,
	swapchain_format:           vk.SurfaceFormatKHR,
	swapchain_extent:           vk.Extent2D,
	swapchain_frame_buffers:    []vk.Framebuffer,
	vert_shader_module:         vk.ShaderModule,
	frag_shader_module:         vk.ShaderModule,
	shader_stages:              [2]vk.PipelineShaderStageCreateInfo,
	render_pass:                vk.RenderPass,
	pipeline_layout:            vk.PipelineLayout,
	pipeline:                   vk.Pipeline,
	command_pool:               vk.CommandPool,
	command_buffers:            [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer,
	image_available_semaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
	render_finished_semaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
	in_flight_fences:           [MAX_FRAMES_IN_FLIGHT]vk.Fence,
	current_frame:              u32,
}

create_instance :: proc(self: ^Renderer) -> vk.Result {
	log.info("Creating Vulkan instance...")
	extensions := slice.clone_to_dynamic(
		glfw.GetRequiredInstanceExtensions(),
		runtime.default_allocator(),
	)
	log.info("Required Vulkan extensions:", len(extensions))
	create_info := vk.InstanceCreateInfo {
		sType               = .INSTANCE_CREATE_INFO,
		pApplicationInfo    = &vk.ApplicationInfo {
			sType = .APPLICATION_INFO,
			pApplicationName = TITLE,
			applicationVersion = vk.MAKE_VERSION(1, 0, 0),
			pEngineName = ENGINE_NAME,
			engineVersion = vk.MAKE_VERSION(1, 0, 0),
			apiVersion = vk.API_VERSION_1_3,
		},
		ppEnabledLayerNames = raw_data(LAYERS),
		enabledLayerCount   = u32(len(LAYERS)),
	}
	when ODIN_OS == .Darwin {
		// Mandatory on macOS
		create_info.flags |= {.ENUMERATE_PORTABILITY_KHR}
		append(&extensions, vk.KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
	}
	when ENABLE_VALIDATION_LAYERS {
		append(&extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
		// Severity based on logger level.
		severity: vk.DebugUtilsMessageSeverityFlagsEXT
		if context.logger.lowest_level <= .Error {
			severity |= {.ERROR}
		}
		if context.logger.lowest_level <= .Warning {
			severity |= {.WARNING}
		}
		if context.logger.lowest_level <= .Info {
			severity |= {.INFO}
		}
		if context.logger.lowest_level <= .Debug {
			severity |= {.VERBOSE}
		}
		dbg_create_info := vk.DebugUtilsMessengerCreateInfoEXT {
			sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			messageSeverity = severity,
			messageType = {.GENERAL, .VALIDATION, .PERFORMANCE, .DEVICE_ADDRESS_BINDING}, // all of them.
			pfnUserCallback = proc "system" (
				severity: vk.DebugUtilsMessageSeverityFlagsEXT,
				types: vk.DebugUtilsMessageTypeFlagsEXT,
				pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
				pUserData: rawptr,
			) -> b32 {
				context = runtime.default_context()
				level: log.Level
				if .ERROR in severity {
					level = .Error
				} else if .WARNING in severity {
					level = .Warning
				} else if .INFO in severity {
					level = .Info
				} else {
					level = .Debug
				}
				log.logf(level, "vulkan[%v]: %s", types, pCallbackData.pMessage)
				return false
			},
		}
		create_info.pNext = &dbg_create_info
	}
	create_info.enabledExtensionCount = u32(len(extensions))
	create_info.ppEnabledExtensionNames = raw_data(extensions)
	vk.CreateInstance(&create_info, nil, &self.instance) or_return
	vk.load_proc_addresses_instance(self.instance)
	when ENABLE_VALIDATION_LAYERS {
		vk.CreateDebugUtilsMessengerEXT(
			self.instance,
			&dbg_create_info,
			nil,
			&self.dbg_messenger,
		) or_return
	}
	return .SUCCESS
}

destroy_instance :: proc(self: ^Renderer) {
	when ENABLE_VALIDATION_LAYERS {
		vk.DestroyDebugUtilsMessengerEXT(self.instance, self.dbg_messenger, nil)
	}
	vk.DestroyInstance(self.instance, nil)
}

create_surface :: proc(self: ^Renderer) -> vk.Result {
	glfw.CreateWindowSurface(self.instance, self.window, nil, &self.surface) or_return
	log.info("vulkan: created surface")
	return .SUCCESS
}

destroy_surface :: proc(self: ^Renderer) {
	vk.DestroySurfaceKHR(self.instance, self.surface, nil)
}

pick_physical_device :: proc(self: ^Renderer) -> vk.Result {
	get_available_extensions :: proc(
		device: vk.PhysicalDevice,
	) -> (
		exts: []vk.ExtensionProperties,
		res: vk.Result,
	) {
		count: u32
		vk.EnumerateDeviceExtensionProperties(device, nil, &count, nil) or_return
		exts = make([]vk.ExtensionProperties, count, context.allocator)
		vk.EnumerateDeviceExtensionProperties(device, nil, &count, raw_data(exts)) or_return
		return
	}
	score_physical_device :: proc(self: ^Renderer, device: vk.PhysicalDevice) -> (score: int) {
		log.infof("vulkan: evaluating device %x", device)
		props: vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(device, &props)
		name := cstring(raw_data(props.deviceName[:]))
		log.infof("vulkan: evaluating device %q", name)
		defer log.infof("vulkan: device %q scored %v", name, score)
		features: vk.PhysicalDeviceFeatures
		vk.GetPhysicalDeviceFeatures(device, &features)
		when ODIN_OS != .Darwin {
			// Apple Silicon somehow doesn't have this
			if !features.geometryShader {
				log.info("vulkan: device does not support geometry shaders")
				return 0
			}
		}
		// Need certain extensions supported.
		{
			exts, result := get_available_extensions(device)
			defer free(raw_data(exts))
			if result != .SUCCESS {
				log.infof("vulkan: enumerate device extension properties failed:", result)
				return 0
			}
			extensions := make(map[cstring]bool)
			defer delete(extensions)
			for &e in exts {
				extensions[cstring(raw_data(e.extensionName[:]))] = true
			}
			log.infof("vulkan: device supports %v extensions", len(extensions))
			for required in REQUIRED_EXTENSIONS {
				if required in extensions {
					continue
				}
				log.infof("vulkan: required extension %q not found", required)
				return 0
			}
			log.info("vulkan: device supports all required extensions")
		}
		{
			support, result := query_swapchain_support(self, device)
			if result != .SUCCESS {
				log.infof("vulkan: query swapchain support failure:", result)
				return 0
			}
			defer {
				free(raw_data(support.formats))
				free(raw_data(support.presentModes))
			}

			// Need at least a format and present mode.
			if len(support.formats) == 0 || len(support.presentModes) == 0 {
				log.info("vulkan: device does not support swapchain")
				return 0
			}
		}

		families := find_queue_families(self, device)
		if _, has_graphics := families.graphics.?; !has_graphics {
			log.info("vulkan: device does not have a graphics queue")
			return 0
		}
		if _, has_present := families.present.?; !has_present {
			log.info("vulkan: device does not have a presentation queue")
			return 0
		}

		// Favor GPUs.
		switch props.deviceType {
		case .DISCRETE_GPU:
			score += 400_000
		case .INTEGRATED_GPU:
			score += 300_000
		case .VIRTUAL_GPU:
			score += 200_000
		case .CPU, .OTHER:
			score += 100_000
		}
		log.infof("vulkan: scored %i based on device type", score, props.deviceType)

		// Maximum texture size.
		score += int(props.limits.maxImageDimension2D)
		log.infof(
			"vulkan: added the max 2D image dimensions (texture size) of %v to the score",
			props.limits.maxImageDimension2D,
		)
		return
	}

	count: u32
	vk.EnumeratePhysicalDevices(self.instance, &count, nil) or_return
	log.infof("vulkan: found %v GPUs", count)
	devices := make([]vk.PhysicalDevice, count, context.allocator)
	defer delete(devices)
	vk.EnumeratePhysicalDevices(self.instance, &count, raw_data(devices)) or_return
	for device in devices {
		log.infof("vulkan: found device %x", device)
	}

	best_score := 0
	for device in devices {
		if score := score_physical_device(self, device); score > best_score {
			self.physical_device = device
			best_score = score
		}
	}

	if best_score <= 0 {
		log.panic("vulkan: no suitable GPU found")
	}
	return .SUCCESS
}

destroy_device :: proc(self: ^Renderer) {
	vk.DestroyDevice(self.device, nil)
}

SwapchainSupport :: struct {
	capabilities: vk.SurfaceCapabilitiesKHR,
	formats:      []vk.SurfaceFormatKHR,
	presentModes: []vk.PresentModeKHR,
}
query_swapchain_support :: proc(
	self: ^Renderer,
	device: vk.PhysicalDevice,
) -> (
	support: SwapchainSupport,
	result: vk.Result,
) {
	// NOTE: looks like a wrong binding with the third arg being a multipointer.
	log.info("vulkan: querying swapchain support for device", device)
	vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(
		device,
		self.surface,
		&support.capabilities,
	) or_return
	log.info("vulkan: got surface capabilities", support.capabilities)
	{
		count: u32
		vk.GetPhysicalDeviceSurfaceFormatsKHR(device, self.surface, &count, nil) or_return

		log.infof("vulkan: found %v surface formats", count)

		support.formats = make([]vk.SurfaceFormatKHR, count, context.allocator)
		vk.GetPhysicalDeviceSurfaceFormatsKHR(
			device,
			self.surface,
			&count,
			raw_data(support.formats),
		) or_return
	}

	{
		count: u32
		vk.GetPhysicalDeviceSurfacePresentModesKHR(device, self.surface, &count, nil) or_return

		support.presentModes = make([]vk.PresentModeKHR, count, context.allocator)
		vk.GetPhysicalDeviceSurfacePresentModesKHR(
			device,
			self.surface,
			&count,
			raw_data(support.presentModes),
		) or_return
	}
	return
}

QueueFamilyIndices :: struct {
	graphics: Maybe(u32),
	present:  Maybe(u32),
}
find_queue_families :: proc(
	self: ^Renderer,
	device: vk.PhysicalDevice,
) -> (
	ids: QueueFamilyIndices,
) {
	count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, nil)

	families := make([]vk.QueueFamilyProperties, count, context.allocator)
	defer delete(families)
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, raw_data(families))

	for family, i in families {
		if .GRAPHICS in family.queueFlags {
			ids.graphics = u32(i)
			log.info("vulkan: found graphics queue family", i)
		}

		supported: b32
		vk.GetPhysicalDeviceSurfaceSupportKHR(device, u32(i), self.surface, &supported)
		if supported {
			ids.present = u32(i)
			log.info("vulkan: found present queue family", i)
		}

		// Found all needed queues?
		_, has_graphics := ids.graphics.?
		_, has_present := ids.present.?
		if has_graphics && has_present {
			break
		}
	}
	return
}

create_logical_device :: proc(self: ^Renderer) -> vk.Result {
	log.info("Creating logical device... with physical device", self.physical_device)
	families := find_queue_families(self, self.physical_device)
	indices_set := make(map[u32]struct {})
	indices_set[families.graphics.?] = {}
	indices_set[families.present.?] = {}
	defer delete(indices_set)

	queue_create_infos := make(
		[dynamic]vk.DeviceQueueCreateInfo,
		0,
		len(indices_set),
		context.allocator,
	)
	defer delete(queue_create_infos)
	for _ in indices_set {
		append(
			&queue_create_infos,
			vk.DeviceQueueCreateInfo {
				sType = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = families.graphics.?,
				queueCount = 1,
				pQueuePriorities = raw_data([]f32{1}),
			}, // Scheduling priority between 0 and 1.
		)
	}

	when ENABLE_VALIDATION_LAYERS {
		layers := []cstring{"VK_LAYER_KHRONOS_validation"}
	} else {
		layers := []cstring{}
	}

	device_create_info := vk.DeviceCreateInfo {
		sType                   = .DEVICE_CREATE_INFO,
		pQueueCreateInfos       = raw_data(queue_create_infos),
		queueCreateInfoCount    = u32(len(queue_create_infos)),
		enabledLayerCount       = u32(len(layers)),
		ppEnabledLayerNames     = raw_data(layers),
		ppEnabledExtensionNames = raw_data(REQUIRED_EXTENSIONS),
		enabledExtensionCount   = u32(len(REQUIRED_EXTENSIONS)),
	}
	vk.CreateDevice(self.physical_device, &device_create_info, nil, &self.device) or_return
	vk.GetDeviceQueue(self.device, families.graphics.?, 0, &self.graphics_queue)
	vk.GetDeviceQueue(self.device, families.present.?, 0, &self.present_queue)
	return .SUCCESS
}

create_swapchain :: proc(self: ^Renderer) -> (result: vk.Result) {
	pick_swapchain_surface_format :: proc(formats: []vk.SurfaceFormatKHR) -> vk.SurfaceFormatKHR {
		for format in formats {
			if format.format == .B8G8R8A8_SRGB && format.colorSpace == .SRGB_NONLINEAR {
				return format
			}
		}
		return formats[0]
	}

	pick_swapchain_present_mode :: proc(modes: []vk.PresentModeKHR) -> vk.PresentModeKHR {
		for mode in modes {
			if mode == .MAILBOX {
				return .MAILBOX
			}
		}
		return .FIFO
	}

	pick_swapchain_extent :: proc(
		window: glfw.WindowHandle,
		capabilities: vk.SurfaceCapabilitiesKHR,
	) -> vk.Extent2D {
		if capabilities.currentExtent.width != max(u32) {
			return capabilities.currentExtent
		}
		width, height := glfw.GetFramebufferSize(window)
		return (vk.Extent2D {
					width = clamp(
						u32(width),
						capabilities.minImageExtent.width,
						capabilities.maxImageExtent.width,
					),
					height = clamp(
						u32(height),
						capabilities.minImageExtent.height,
						capabilities.maxImageExtent.height,
					),
				})
	}
	families := find_queue_families(self, self.physical_device)

	// Setup swapchain.
	{
		support := query_swapchain_support(self, self.physical_device) or_return
		defer {
			free(raw_data(support.formats))
			free(raw_data(support.presentModes))
		}
		surface_format := pick_swapchain_surface_format(support.formats)
		present_mode := pick_swapchain_present_mode(support.presentModes)
		extent := pick_swapchain_extent(self.window, support.capabilities)

		self.swapchain_format = surface_format
		self.swapchain_extent = extent

		image_count: u32
		unlimitted := support.capabilities.maxImageCount == 0
		if unlimitted {
			image_count = support.capabilities.minImageCount + 1
		} else {
			image_count = min(
				support.capabilities.maxImageCount,
				support.capabilities.minImageCount + 1,
			)
		}

		create_info := vk.SwapchainCreateInfoKHR {
			sType            = .SWAPCHAIN_CREATE_INFO_KHR,
			surface          = self.surface,
			minImageCount    = image_count,
			imageFormat      = surface_format.format,
			imageColorSpace  = surface_format.colorSpace,
			imageExtent      = extent,
			imageArrayLayers = 1,
			imageUsage       = {.COLOR_ATTACHMENT},
			preTransform     = support.capabilities.currentTransform,
			compositeAlpha   = {.OPAQUE},
			presentMode      = present_mode,
			clipped          = true,
		}

		if families.graphics != families.present {
			create_info.imageSharingMode = .CONCURRENT
			create_info.queueFamilyIndexCount = 2
			create_info.pQueueFamilyIndices = raw_data(
				[]u32{families.graphics.?, families.present.?},
			)
		}

		vk.CreateSwapchainKHR(self.device, &create_info, nil, &self.swapchain) or_return
	}

	// Setup swapchain images.
	{
		count: u32
		vk.GetSwapchainImagesKHR(self.device, self.swapchain, &count, nil) or_return
		log.infof("vulkan: found %v swapchain images", count)
		self.swapchain_images = make([]vk.Image, count)
		self.swapchain_views = make([]vk.ImageView, count)
		vk.GetSwapchainImagesKHR(
			self.device,
			self.swapchain,
			&count,
			raw_data(self.swapchain_images),
		) or_return

		for image, i in self.swapchain_images {
			create_info := vk.ImageViewCreateInfo {
				sType = .IMAGE_VIEW_CREATE_INFO,
				image = image,
				viewType = .D2,
				format = self.swapchain_format.format,
				subresourceRange = {aspectMask = {.COLOR}, levelCount = 1, layerCount = 1},
			}
			vk.CreateImageView(self.device, &create_info, nil, &self.swapchain_views[i]) or_return
			log.infof("vulkan: created image view %v", i)
		}
	}
	return .SUCCESS
}

destroy_swapchain :: proc(self: ^Renderer) {
	for view in self.swapchain_views {
		vk.DestroyImageView(self.device, view, nil)
	}
	delete(self.swapchain_views)
	delete(self.swapchain_images)
	vk.DestroySwapchainKHR(self.device, self.swapchain, nil)
}

create_shader_module :: proc(
	self: ^Renderer,
	code: []u8,
) -> (
	module: vk.ShaderModule,
	result: vk.Result,
) {
	create_info := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(code),
		pCode    = raw_data(slice.reinterpret([]u32, code)),
	}
	vk.CreateShaderModule(self.device, &create_info, nil, &module) or_return
	return
}

load_shader :: proc(self: ^Renderer) -> vk.Result {
	self.vert_shader_module = create_shader_module(self, SHADER_VERT) or_return
	self.shader_stages[0] = vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.VERTEX},
		module = self.vert_shader_module,
		pName  = "main",
	}

	self.frag_shader_module = create_shader_module(self, SHADER_FRAG) or_return
	self.shader_stages[1] = vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.FRAGMENT},
		module = self.frag_shader_module,
		pName  = "main",
	}
	return .SUCCESS
}

destroy_shader :: proc(self: ^Renderer) {
	vk.DestroyShaderModule(self.device, self.vert_shader_module, nil)
	vk.DestroyShaderModule(self.device, self.frag_shader_module, nil)
}

create_render_pass :: proc(self: ^Renderer) -> vk.Result {
	color_attachment := vk.AttachmentDescription {
		format         = self.swapchain_format.format,
		samples        = {._1},
		loadOp         = .CLEAR,
		storeOp        = .STORE,
		stencilLoadOp  = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout  = .UNDEFINED,
		finalLayout    = .PRESENT_SRC_KHR,
	}
	color_attachment_ref := vk.AttachmentReference {
		attachment = 0,
		layout     = .COLOR_ATTACHMENT_OPTIMAL,
	}
	subpass := vk.SubpassDescription {
		pipelineBindPoint    = .GRAPHICS,
		colorAttachmentCount = 1,
		pColorAttachments    = &color_attachment_ref,
	}
	dependency := vk.SubpassDependency {
		srcSubpass    = vk.SUBPASS_EXTERNAL,
		dstSubpass    = 0,
		srcStageMask  = {.COLOR_ATTACHMENT_OUTPUT},
		srcAccessMask = {},
		dstStageMask  = {.COLOR_ATTACHMENT_OUTPUT},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
	}
	render_pass := vk.RenderPassCreateInfo {
		sType           = .RENDER_PASS_CREATE_INFO,
		attachmentCount = 1,
		pAttachments    = &color_attachment,
		subpassCount    = 1,
		pSubpasses      = &subpass,
		dependencyCount = 1,
		pDependencies   = &dependency,
	}
	return vk.CreateRenderPass(self.device, &render_pass, nil, &self.render_pass)
}

destroy_render_pass :: proc(self: ^Renderer) {
	vk.DestroyRenderPass(self.device, self.render_pass, nil)
}

create_framebuffers :: proc(self: ^Renderer) -> vk.Result {
	self.swapchain_frame_buffers = make([]vk.Framebuffer, len(self.swapchain_views))
	for view, i in self.swapchain_views {
		attachments := []vk.ImageView{view}

		frame_buffer := vk.FramebufferCreateInfo {
			sType           = .FRAMEBUFFER_CREATE_INFO,
			renderPass      = self.render_pass,
			attachmentCount = 1,
			pAttachments    = raw_data(attachments),
			width           = self.swapchain_extent.width,
			height          = self.swapchain_extent.height,
			layers          = 1,
		}
		vk.CreateFramebuffer(
			self.device,
			&frame_buffer,
			nil,
			&self.swapchain_frame_buffers[i],
		) or_return
	}
	return .SUCCESS
}

destroy_framebuffers :: proc(self: ^Renderer) {
	for frame_buffer in self.swapchain_frame_buffers {
		vk.DestroyFramebuffer(self.device, frame_buffer, nil)
	}
	delete(self.swapchain_frame_buffers)
}

create_pipeline :: proc(self: ^Renderer) -> vk.Result {
	dynamic_states := []vk.DynamicState{.VIEWPORT, .SCISSOR}
	dynamic_state := vk.PipelineDynamicStateCreateInfo {
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = 2,
		pDynamicStates    = raw_data(dynamic_states),
	}

	vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
		sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	}

	input_assembly := vk.PipelineInputAssemblyStateCreateInfo {
		sType    = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology = .TRIANGLE_LIST,
	}

	viewport_state := vk.PipelineViewportStateCreateInfo {
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		scissorCount  = 1,
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo {
		sType       = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		polygonMode = .FILL,
		lineWidth   = 1,
		cullMode    = {.BACK},
		frontFace   = .CLOCKWISE,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo {
		sType                = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		rasterizationSamples = {._1},
		minSampleShading     = 1,
	}

	color_blend_attachment := vk.PipelineColorBlendAttachmentState {
		colorWriteMask = {.R, .G, .B, .A},
	}

	color_blending := vk.PipelineColorBlendStateCreateInfo {
		sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		attachmentCount = 1,
		pAttachments    = &color_blend_attachment,
	}

	pipeline_layout := vk.PipelineLayoutCreateInfo {
		sType = .PIPELINE_LAYOUT_CREATE_INFO,
	}
	vk.CreatePipelineLayout(self.device, &pipeline_layout, nil, &self.pipeline_layout) or_return

	pipeline := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		stageCount          = 2,
		pStages             = &self.shader_stages[0],
		pVertexInputState   = &vertex_input_info,
		pInputAssemblyState = &input_assembly,
		pViewportState      = &viewport_state,
		pRasterizationState = &rasterizer,
		pMultisampleState   = &multisampling,
		pColorBlendState    = &color_blending,
		pDynamicState       = &dynamic_state,
		layout              = self.pipeline_layout,
		renderPass          = self.render_pass,
		subpass             = 0,
		basePipelineIndex   = -1,
	}
	return vk.CreateGraphicsPipelines(self.device, 0, 1, &pipeline, nil, &self.pipeline)
}

destroy_pipeline :: proc(self: ^Renderer) {
	vk.DestroyPipelineLayout(self.device, self.pipeline_layout, nil)
	vk.DestroyPipeline(self.device, self.pipeline, nil)
}

create_command_pool :: proc(self: ^Renderer) -> vk.Result {
	families := find_queue_families(self, self.physical_device)
	pool_info := vk.CommandPoolCreateInfo {
		sType            = .COMMAND_POOL_CREATE_INFO,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = families.graphics.?,
	}
	vk.CreateCommandPool(self.device, &pool_info, nil, &self.command_pool) or_return
	alloc_info := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = self.command_pool,
		level              = .PRIMARY,
		commandBufferCount = MAX_FRAMES_IN_FLIGHT,
	}
	return vk.AllocateCommandBuffers(self.device, &alloc_info, &self.command_buffers[0])
}

destroy_command_pool :: proc(self: ^Renderer) {
	vk.DestroyCommandPool(self.device, self.command_pool, nil)
}

create_semaphores :: proc(self: ^Renderer) -> vk.Result {
	sem_info := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
	}
	fence_info := vk.FenceCreateInfo {
		sType = .FENCE_CREATE_INFO,
		flags = {.SIGNALED},
	}
	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		vk.CreateSemaphore(
			self.device,
			&sem_info,
			nil,
			&self.image_available_semaphores[i],
		) or_return
		vk.CreateSemaphore(
			self.device,
			&sem_info,
			nil,
			&self.render_finished_semaphores[i],
		) or_return
		vk.CreateFence(self.device, &fence_info, nil, &self.in_flight_fences[i]) or_return
	}
	return .SUCCESS
}

detroy_semaphores :: proc(self: ^Renderer) {
	for sem in self.image_available_semaphores {
		vk.DestroySemaphore(self.device, sem, nil)
	}
	for sem in self.render_finished_semaphores {
		vk.DestroySemaphore(self.device, sem, nil)
	}
	for fence in self.in_flight_fences {
		vk.DestroyFence(self.device, fence, nil)
	}
}

render :: proc(self: ^Renderer) -> vk.Result {
	// Wait for previous frame.
	vk.WaitForFences(
		self.device,
		1,
		&self.in_flight_fences[self.current_frame],
		true,
		max(u64),
	) or_return

	// Acquire an image from the swapchain.
	image_index: u32
	acquire_result := vk.AcquireNextImageKHR(
		self.device,
		self.swapchain,
		max(u64),
		self.image_available_semaphores[self.current_frame],
		0,
		&image_index,
	)
	#partial switch acquire_result {
	case .ERROR_OUT_OF_DATE_KHR:
		recreate_swapchain(self)
		return .SUCCESS
	case .SUCCESS, .SUBOPTIMAL_KHR:
	case:
		log.panicf("vulkan: acquire next image failure: %v", acquire_result)
	}
	vk.ResetFences(self.device, 1, &self.in_flight_fences[self.current_frame]) or_return
	vk.ResetCommandBuffer(self.command_buffers[self.current_frame], {}) or_return
	record_command_buffer(self, image_index) or_return

	// Submit.
	submit_info := vk.SubmitInfo {
		sType                = .SUBMIT_INFO,
		waitSemaphoreCount   = 1,
		pWaitSemaphores      = &self.image_available_semaphores[self.current_frame],
		pWaitDstStageMask    = &vk.PipelineStageFlags{.COLOR_ATTACHMENT_OUTPUT},
		commandBufferCount   = 1,
		pCommandBuffers      = &self.command_buffers[self.current_frame],
		signalSemaphoreCount = 1,
		pSignalSemaphores    = &self.render_finished_semaphores[self.current_frame],
	}
	vk.QueueSubmit(
		self.graphics_queue,
		1,
		&submit_info,
		self.in_flight_fences[self.current_frame],
	) or_return
	// Present.
	present_info := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &self.render_finished_semaphores[self.current_frame],
		swapchainCount     = 1,
		pSwapchains        = &self.swapchain,
		pImageIndices      = &image_index,
	}
	present_result := vk.QueuePresentKHR(self.present_queue, &present_info)
	switch {
	case present_result == .ERROR_OUT_OF_DATE_KHR || present_result == .SUBOPTIMAL_KHR:
		recreate_swapchain(self)
	case present_result == .SUCCESS:
	case:
		log.panicf("vulkan: present failure: %v", present_result)
	}
	self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT
	return .SUCCESS
}

record_command_buffer :: proc(self: ^Renderer, image_index: u32) -> vk.Result {
	command_buffer := self.command_buffers[self.current_frame]
	begin_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
	}
	vk.BeginCommandBuffer(command_buffer, &begin_info) or_return
	clear_color := vk.ClearValue {
		color = vk.ClearColorValue{float32 = {0.0117, 0.0117, 0.0179, 1.0}},
	}

	render_pass_info := vk.RenderPassBeginInfo {
		sType = .RENDER_PASS_BEGIN_INFO,
		renderPass = self.render_pass,
		framebuffer = self.swapchain_frame_buffers[image_index],
		renderArea = {extent = self.swapchain_extent},
		clearValueCount = 1,
		pClearValues = &clear_color,
	}
	vk.CmdBeginRenderPass(command_buffer, &render_pass_info, .INLINE)
	vk.CmdBindPipeline(command_buffer, .GRAPHICS, self.pipeline)
	viewport := vk.Viewport {
		width    = f32(self.swapchain_extent.width),
		height   = f32(self.swapchain_extent.height),
		maxDepth = 1.0,
	}
	vk.CmdSetViewport(command_buffer, 0, 1, &viewport)
	scissor := vk.Rect2D {
		extent = self.swapchain_extent,
	}
	vk.CmdSetScissor(command_buffer, 0, 1, &scissor)
	vk.CmdDraw(command_buffer, 6, 1, 0, 0)
	vk.CmdEndRenderPass(command_buffer)
	vk.EndCommandBuffer(command_buffer) or_return
	return .SUCCESS
}

recreate_swapchain :: proc(self: ^Renderer) {
	vk.DeviceWaitIdle(self.device)
	destroy_framebuffers(self)
	destroy_swapchain(self)
	create_swapchain(self)
	create_framebuffers(self)
}

init :: proc(self: ^Renderer, window: glfw.WindowHandle) -> vk.Result {
	self.window = window
	vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))
	create_instance(self) or_return
	create_surface(self) or_return
	pick_physical_device(self) or_return
	create_logical_device(self) or_return
	create_swapchain(self) or_return
	load_shader(self) or_return
	create_render_pass(self) or_return
	create_framebuffers(self) or_return
	create_pipeline(self) or_return
	create_command_pool(self) or_return
	create_semaphores(self) or_return
	return .SUCCESS
}

destroy :: proc(self: ^Renderer) {
	vk.DeviceWaitIdle(self.device)
	detroy_semaphores(self)
	destroy_command_pool(self)
	destroy_pipeline(self)
	destroy_render_pass(self)
	destroy_shader(self)
	destroy_framebuffers(self)
	destroy_swapchain(self)
	destroy_device(self)
	destroy_surface(self)
	destroy_instance(self)
}
