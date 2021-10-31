#![cfg(windows)]

use imgui::internal::RawWrapper;
use imgui::{
    BackendFlags, DrawCmd, DrawCmdParams, DrawData, DrawIdx, DrawVert, TextureId, Textures,
};
use log::info;
use memoffset::offset_of;

use core::num::NonZeroI32;
use core::ptr;
use core::slice;
use core::{mem, ops};

use rusty_d3d12::*;

use thiserror::Error;

// IDR is for imgui-d3d12-renderer
#[derive(Error, Debug)]
pub enum IDRError {}

pub type IDRResult<T> = Result<T, IDRError>;

const FONT_TEX_ID: usize = !0;

const VERTEX_BUF_ADD_CAPACITY: usize = 5000;
const INDEX_BUF_ADD_CAPACITY: usize = 10000;

#[repr(C)]
struct VertexConstantBuffer {
    mvp: [[f32; 4]; 4],
}

/// A D3D12 renderer for (Imgui-rs)[https://docs.rs/imgui/*/imgui/].
#[derive(Debug)]
pub struct Renderer {
    // device: ComPtr<ID3D11Device>,
// context: ComPtr<ID3D11DeviceContext>,
// factory: ComPtr<IDXGIFactory>,
// vertex_shader: ComPtr<ID3D11VertexShader>,
// pixel_shader: ComPtr<ID3D11PixelShader>,
// input_layout: ComPtr<ID3D11InputLayout>,
// constant_buffer: ComPtr<ID3D11Buffer>,
// blend_state: ComPtr<ID3D11BlendState>,
// rasterizer_state: ComPtr<ID3D11RasterizerState>,
// depth_stencil_state: ComPtr<ID3D11DepthStencilState>,
// font_resource_view: ComPtr<ID3D11ShaderResourceView>,
// font_sampler: ComPtr<ID3D11SamplerState>,
// vertex_buffer: Buffer,
// index_buffer: Buffer,
// textures: Textures<ComPtr<ID3D11ShaderResourceView>>,
}

fn create_shaders() -> IDRResult<(Vec<u8>, Vec<u8>)> {
    let vertex_shader = compile_shader(
        "VertexShader",
        r"#
cbuffer vertexBuffer: register(b0) {
            float4x4 ProjectionMatrix;
        };

        struct VS_INPUT {
            float2 pos: POSITION;
            float4 col: COLOR0;
            float2 uv: TEXCOORD0;
        };

        struct PS_INPUT {
            float4 pos: SV_POSITION;
            float4 col: COLOR0;
            float2 uv: TEXCOORD0;
        };

        PS_INPUT main(VS_INPUT input) {
            PS_INPUT output;
            output.pos = mul(ProjectionMatrix, float4(input.pos.xy, 0.f, 1.f));
            output.col = input.col;
            output.uv = input.uv;
            return output;
        }
#",
        "main",
        "vs_6_0",
        &[],
        &[],
    )?;

    let pixel_shader = compile_shader(
        "PixelShader",
        r"#
struct PS_INPUT {
    float4 pos: SV_POSITION;
    float4 col: COLOR0;
    float2 uv: TEXCOORD0;
};

sampler sampler0;
Texture2D texture0;

float4 main(PS_INPUT input): SV_Target {
    float4 out_col = input.col * texture0.Sample(sampler0, input.uv);
    return out_col;
}
#",
        "main",
        "ps_6_0",
        &[],
        &[],
    )
    .expect("Cannot compile pixel shader")?;

    Ok((vertex_shader, pixel_shader))
}

fn create_pipeline_state(
    input_layout: Vec<InputElementDesc>,
    root_signature: &RootSignature,
    vertex_shader: Vec<u8>,
    pixel_shader: Vec<u8>,
    device: &Device,
) -> IDRResult<PipelineState> {
    let vs_bytecode = ShaderBytecode::from_bytes(&vertex_shader);
    let ps_bytecode = ShaderBytecode::from_bytes(&pixel_shader);

    let input_layout = InputLayoutDesc::default().from_input_elements(&input_layout);
    let pso_desc = GraphicsPipelineStateDesc::default()
        .set_input_layout(&input_layout)
        .set_root_signature(root_signature)
        .set_vs_bytecode(&vs_bytecode)
        .set_ps_bytecode(&ps_bytecode)
        .set_rasterizer_state(
            &RasterizerDesc::default().set_fill_mode(FillMode::Solid).set_depth_clip_enable(true),
        )
        .set_blend_state(
            &BlendDesc::default().set_render_targets(&[RenderTargetBlendDesc::default()
                .set_blend_enable(true)
                .set_src_blend(Blend::SrcAlpha)
                .set_dest_blend(Blend::InvSrcAlpha)
                .set_blend_op(BlendOp::Add)
                .set_src_blend_alpha(Blend::InvDestAlpha)
                .set_dest_blend_alpha(Blend::One)
                .set_blend_op_alpha(BlendOp::Add)
                .set_render_target_write_mask(ColorWriteEnable::EnableAll)]),
        )
        .set_depth_stencil_state(&DepthStencilDesc::default())
        .set_primitive_topology_type(PrimitiveTopologyType::Triangle)
        .set_rtv_formats(&[Format::R8G8B8A8_UNorm])
        .set_dsv_format(Format::D32_Float);

    device.create_graphics_pipeline_state(&pso_desc)
}

fn create_input_layout() -> Vec<InputElementDesc<'static>> {
    vec![
        InputElementDesc::default()
            // ToDo: "POSITION\0" on lib side would allow to get rid of allocations
            .set_name("POSITION")
            .unwrap()
            .set_format(Format::R32G32B32_Float)
            .set_input_slot(0)
            .set_offset(Bytes::from(offset_of!(DrawVert, pos))),
        InputElementDesc::default()
            .set_name("TEXCOORD")
            .unwrap()
            .set_format(Format::R32G32_Float)
            .set_input_slot(0)
            .set_offset(Bytes::from(offset_of!(DrawVert, uv))),
        InputElementDesc::default()
            .set_name("COLOR")
            .unwrap()
            .set_format(Format::R8G8B8A8_UNorm)
            .set_input_slot(0)
            .set_offset(Bytes::from(offset_of!(DrawVert, col))),
    ]
}

fn setup_root_signature(device: &Device) -> IDRResult<RootSignature> {
    let ranges = [
        DescriptorRange::default()
            .set_range_type(DescriptorRangeType::Cbv)
            .set_num_descriptors(1)
            .set_flags(DescriptorRangeFlags::DataVolatile),
        DescriptorRange::default()
            .set_range_type(DescriptorRangeType::Srv)
            .set_num_descriptors(1)
            .set_flags(DescriptorRangeFlags::DataVolatile),
    ];

    let static_sampler_desc = StaticSamplerDesc::default()
        .set_filter(Filter::MinMagMipLinear)
        .set_address_u(TextureAddressMode::Wrap)
        .set_address_v(TextureAddressMode::Wrap)
        .set_address_w(TextureAddressMode::Wrap)
        .set_comparison_func(ComparisonFunc::Always)
        .set_border_color(StaticBorderColor::TransparentBlack)
        .set_shader_visibility(ShaderVisibility::Pixel);

    let descriptor_table = RootDescriptorTable::default().set_descriptor_ranges(&ranges);

    let root_parameters = vec![RootParameter::default()
        .new_descriptor_table(&descriptor_table)
        .set_shader_visibility(ShaderVisibility::All)];
    let root_signature_desc = VersionedRootSignatureDesc::default().set_desc_1_1(
        &RootSignatureDesc::default()
            .set_parameters(&root_parameters)
            .set_static_samplers(slice::from_ref(&static_sampler_desc))
            .set_flags(RootSignatureFlags::AllowInputAssemblerInputLayout),
    );

    let (serialized_signature, serialization_result) =
        RootSignature::serialize_versioned(&root_signature_desc);
    assert!(serialization_result.is_ok(), "Result: {}", &serialization_result.err().unwrap());

    device.create_root_signature(0, &ShaderBytecode::from_bytes(serialized_signature.get_buffer()))
}

fn create_font_texture(
    mut fonts: imgui::FontAtlasRefMut<'_>,
    device: &Device,
    font_tex_cpu_descriptor_handle: CpuDescriptorHandle,
)
//-> IDRResult<(ComPtr<ID3D11ShaderResourceView>, ComPtr<ID3D11SamplerState>)> {
{
    let fa_tex = fonts.build_rgba32_texture();

    let texture_desc = ResourceDesc::default()
        .set_dimension(ResourceDimension::Texture2D)
        .set_width(fa_tex.width)
        .set_height(fa_tex.height)
        .set_mip_levels(1)
        .set_format(Format::R8G8B8A8_UNorm);

    let (staging_resource, texture_resource) = upload_texture(device, &texture_desc, fa_tex.data);

    device.create_shader_resource_view(&texture_resource, None, font_tex_cpu_descriptor_handle);

    fonts.tex_id = TextureId::from(FONT_TEX_ID);

    let desc = D3D11_SAMPLER_DESC {
        Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D11_TEXTURE_ADDRESS_WRAP,
        AddressV: D3D11_TEXTURE_ADDRESS_WRAP,
        AddressW: D3D11_TEXTURE_ADDRESS_WRAP,
        MipLODBias: 0.0,
        MaxAnisotropy: 0,
        ComparisonFunc: D3D11_COMPARISON_ALWAYS,
        BorderColor: [0.0; 4],
        MinLOD: 0.0,
        MaxLOD: 0.0,
    };
    let font_sampler =
        com_ptr_from_fn(|font_sampler| device.CreateSamplerState(&desc, font_sampler))?;
    Ok((font_texture_view, font_sampler))
}

fn upload_texture(
    device: &Device,
    texture_desc: &ResourceDesc,
    init_data: &[u8],
) -> IDRResult<(Resource, Resource)> {
    let staging_resource = device.create_committed_resource(
        &HeapProperties::default().set_heap_type(HeapType::Upload),
        HeapFlags::None,
        texture_desc,
        ResourceStates::CopySource,
        None,
    )?;

    let staging_data = staging_resource.map(0, None)?;

    unsafe {
        std::ptr::copy_nonoverlapping(init_data.as_ptr(), staging_data, init_data.len());
    }

    staging_resource.unmap(0, None);

    let texture_resource = device.create_committed_resource(
        &HeapProperties::default().set_heap_type(HeapType::Default),
        HeapFlags::None,
        texture_desc,
        ResourceStates::CopyDest,
        None,
    )?;

    let command_queue = device.create_command_queue(
        &CommandQueueDesc::default()
            .set_queue_type(CommandListType::Direct)
            .set_flags(CommandQueueFlags::None),
    )?;

    let command_allocator = device.create_command_allocator(CommandListType::Direct)?;

    let command_list =
        device.create_command_list(CommandListType::Dirce, &command_allocator, None)?;

    let mut fence_value = 0;
    let fence = device.create_fence(fence_value, FenceFlags::None)?;
    let event = Win32Event::default();

    command_list.copy_resource(&texture_resource, &staging_resource);

    command_list.resource_barrier(std::slice::from_ref(&ResourceBarrier::new_transition(
        &ResourceTransitionBarrier::default()
            .set_resource(&texture_resource)
            .set_state_before(ResourceStates::CopyDest)
            .set_state_after(ResourceStates::PixelShaderResource),
    )));

    command_list.close()?;
    command_queue.execute_command_lists(slice::from_ref(&command_list));

    fence_value += 1;
    command_queue.signal(&fence, fence_value);

    fence.set_event_on_completion(fence_value, &event);
    event.wait(None);

    Ok((staging_resource, texture_resource))
}

impl Renderer {
    pub fn new(
        im_ctx: &mut imgui::Context,
        device: &Device,
        frame_count: usize,
        font_tex_cpu_descriptor_handle: CpuDescriptorHandle,
        font_tex_gpu_descriptor_handle: GpuDescriptorHandle,
    ) -> IDRResult<Self> {
        let (vertex_shader, pixel_shader) = create_shaders();

        let input_layout = create_input_layout();
        let root_signature = setup_root_signature(device);
        let pipeline_state = create_pipeline_state(
            input_layout,
            &root_signature,
            vertex_shader,
            pixel_shader,
            device,
        );

        let (font_resource_view, font_sampler) = create_font_texture(im_ctx.fonts(), &device)?;
        let vertex_buffer = Self::create_vertex_buffer(&device, 0)?;
        let index_buffer = Self::create_index_buffer(&device, 0)?;
        let context = {
            let mut context = ptr::null_mut();
            device.GetImmediateContext(&mut context);
            ComPtr::from_raw(context)
        };
        im_ctx.io_mut().backend_flags |= BackendFlags::RENDERER_HAS_VTX_OFFSET;
        im_ctx.set_renderer_name(Some(
            concat!("imgui_d3d12_renderer@", env!("CARGO_PKG_VERSION")).to_owned(),
        ));

        Ok(Renderer {
            device,
            context,
            factory,
            vertex_shader,
            pixel_shader,
            input_layout,
            constant_buffer,
            blend_state,
            rasterizer_state,
            depth_stencil_state,
            font_resource_view,
            font_sampler,
            vertex_buffer,
            index_buffer,
            textures: Textures::new(),
        })
    }

    // /// The textures registry of this renderer.
    // ///
    // /// The texture slot at !0 is reserved for the font texture, therefore the
    // /// renderer will ignore any texture inserted into said slot.
    // #[inline]
    // pub fn textures_mut(&mut self) -> &mut Textures<ComPtr<ID3D11ShaderResourceView>> {
    //     &mut self.textures
    // }

    /// The textures registry of this renderer.
    // #[inline]
    // pub fn textures(&self) -> &Textures<ComPtr<ID3D11ShaderResourceView>> {
    //     &self.textures
    // }

    /// Renders the given [`Ui`] with this renderer.
    ///
    /// Should the [`DrawData`] contain an invalid texture index the renderer
    /// will return `DXGI_ERROR_INVALID_CALL` and immediately stop rendering.
    ///
    /// [`Ui`]: https://docs.rs/imgui/*/imgui/struct.Ui.html
    pub fn render(&mut self, draw_data: &DrawData) -> HResult<()> {
        if draw_data.display_size[0] <= 0.0 || draw_data.display_size[1] <= 0.0 {
            return Ok(());
        }

        unsafe {
            if self.vertex_buffer.len() < draw_data.total_vtx_count as usize {
                self.vertex_buffer =
                    Self::create_vertex_buffer(&self.device, draw_data.total_vtx_count as usize)?;
            }
            if self.index_buffer.len() < draw_data.total_idx_count as usize {
                self.index_buffer =
                    Self::create_index_buffer(&self.device, draw_data.total_idx_count as usize)?;
            }

            let _state_guard = StateBackup::backup(&self.context);

            self.write_buffers(draw_data)?;
            self.setup_render_state(draw_data);
            self.render_impl(draw_data)?;
        }
        Ok(())
    }

    unsafe fn render_impl(&self, draw_data: &DrawData) -> HResult<()> {
        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;
        let mut vertex_offset = 0;
        let mut index_offset = 0;
        let mut last_tex = TextureId::from(FONT_TEX_ID);
        self.context.PSSetShaderResources(0, 1, &self.font_resource_view.as_raw());
        for draw_list in draw_data.draw_lists() {
            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements {
                        count,
                        cmd_params: DrawCmdParams { clip_rect, texture_id, .. },
                    } => {
                        if texture_id != last_tex {
                            let texture = if texture_id.id() == FONT_TEX_ID {
                                &self.font_resource_view
                            } else {
                                self.textures.get(texture_id).ok_or(DXGI_ERROR_INVALID_CALL)?
                            };
                            self.context.PSSetShaderResources(0, 1, &texture.as_raw());
                            last_tex = texture_id;
                        }

                        let r = D3D11_RECT {
                            left: ((clip_rect[0] - clip_off[0]) * clip_scale[0]) as i32,
                            top: ((clip_rect[1] - clip_off[1]) * clip_scale[1]) as i32,
                            right: ((clip_rect[2] - clip_off[0]) * clip_scale[0]) as i32,
                            bottom: ((clip_rect[3] - clip_off[1]) * clip_scale[1]) as i32,
                        };
                        self.context.RSSetScissorRects(1, &r);
                        self.context.DrawIndexed(
                            count as u32,
                            index_offset as u32,
                            vertex_offset as i32,
                        );
                        index_offset += count;
                    }
                    DrawCmd::ResetRenderState => self.setup_render_state(draw_data),
                    DrawCmd::RawCallback { callback, raw_cmd } => {
                        callback(draw_list.raw(), raw_cmd)
                    }
                }
            }
            vertex_offset += draw_list.vtx_buffer().len();
        }
        Ok(())
    }

    unsafe fn setup_render_state(&self, draw_data: &DrawData) {
        let ctx = &*self.context;

        let vp = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: draw_data.display_size[0] * draw_data.framebuffer_scale[0],
            Height: draw_data.display_size[1] * draw_data.framebuffer_scale[1],
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        ctx.RSSetViewports(1, &vp);

        let stride = mem::size_of::<DrawVert>() as u32;
        ctx.IASetInputLayout(self.input_layout.as_raw());
        ctx.IASetVertexBuffers(0, 1, &self.vertex_buffer.as_raw(), &stride, &0);
        ctx.IASetIndexBuffer(
            self.index_buffer.as_raw(),
            if mem::size_of::<DrawIdx>() == 2 {
                DXGI_FORMAT_R16_UINT
            } else {
                DXGI_FORMAT_R32_UINT
            },
            0,
        );
        ctx.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx.VSSetShader(self.vertex_shader.as_raw(), ptr::null(), 0);
        ctx.VSSetConstantBuffers(0, 1, &self.constant_buffer.as_raw());
        ctx.PSSetShader(self.pixel_shader.as_raw(), ptr::null(), 0);
        ctx.PSSetSamplers(0, 1, &self.font_sampler.as_raw());
        ctx.GSSetShader(ptr::null_mut(), ptr::null(), 0);
        ctx.HSSetShader(ptr::null_mut(), ptr::null(), 0);
        ctx.DSSetShader(ptr::null_mut(), ptr::null(), 0);
        ctx.CSSetShader(ptr::null_mut(), ptr::null(), 0);

        let blend_factor = [0.0; 4];
        ctx.OMSetBlendState(self.blend_state.as_raw(), &blend_factor, 0xFFFFFFFF);
        ctx.OMSetDepthStencilState(self.depth_stencil_state.as_raw(), 0);
        ctx.RSSetState(self.rasterizer_state.as_raw());
    }

    unsafe fn create_vertex_buffer(
        device: &ComPtr<ID3D11Device>,
        vtx_count: usize,
    ) -> HResult<Buffer> {
        let len = vtx_count + VERTEX_BUF_ADD_CAPACITY;
        let desc = D3D11_BUFFER_DESC {
            ByteWidth: (len * mem::size_of::<DrawVert>()) as u32,
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_VERTEX_BUFFER,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE,
            MiscFlags: 0,
            StructureByteStride: 0,
        };
        com_ptr_from_fn(|vertex_buffer| device.CreateBuffer(&desc, ptr::null(), vertex_buffer))
            .map(|vb| Buffer(vb, len))
    }

    unsafe fn create_index_buffer(
        device: &ComPtr<ID3D11Device>,
        idx_count: usize,
    ) -> HResult<Buffer> {
        let len = idx_count + INDEX_BUF_ADD_CAPACITY;
        let desc = D3D11_BUFFER_DESC {
            ByteWidth: (len * mem::size_of::<DrawIdx>()) as u32,
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_INDEX_BUFFER,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE,
            MiscFlags: 0,
            StructureByteStride: 0,
        };
        com_ptr_from_fn(|index_buffer| device.CreateBuffer(&desc, ptr::null(), index_buffer))
            .map(|ib| Buffer(ib, len))
    }

    unsafe fn write_buffers(&self, draw_data: &DrawData) -> HResult<()> {
        let mut vtx_resource = mem::MaybeUninit::zeroed();
        let mut idx_resource = mem::MaybeUninit::zeroed();
        hresult(self.context.Map(
            self.vertex_buffer.as_raw().cast(),
            0,
            D3D11_MAP_WRITE_DISCARD,
            0,
            vtx_resource.as_mut_ptr(),
        ))?;
        if let e @ Err(_) = hresult(self.context.Map(
            self.index_buffer.as_raw().cast(),
            0,
            D3D11_MAP_WRITE_DISCARD,
            0,
            idx_resource.as_mut_ptr(),
        )) {
            self.context.Unmap(self.vertex_buffer.as_raw().cast(), 0);
            e?;
        }
        let vtx_resource = vtx_resource.assume_init();
        let idx_resource = idx_resource.assume_init();

        let mut vtx_dst = slice::from_raw_parts_mut(
            vtx_resource.pData.cast::<DrawVert>(),
            draw_data.total_vtx_count as usize,
        );
        let mut idx_dst = slice::from_raw_parts_mut(
            idx_resource.pData.cast::<DrawIdx>(),
            draw_data.total_idx_count as usize,
        );
        for (vbuf, ibuf) in
            draw_data.draw_lists().map(|draw_list| (draw_list.vtx_buffer(), draw_list.idx_buffer()))
        {
            vtx_dst[..vbuf.len()].copy_from_slice(vbuf);
            idx_dst[..ibuf.len()].copy_from_slice(ibuf);
            vtx_dst = &mut vtx_dst[vbuf.len()..];
            idx_dst = &mut idx_dst[ibuf.len()..];
        }

        self.context.Unmap(self.vertex_buffer.as_raw().cast(), 0);
        self.context.Unmap(self.index_buffer.as_raw().cast(), 0);

        // constant buffer
        let mut mapped_resource = mem::MaybeUninit::zeroed();
        hresult(self.context.Map(
            com_ref_cast(&self.constant_buffer).as_raw(),
            0,
            D3D11_MAP_WRITE_DISCARD,
            0,
            mapped_resource.as_mut_ptr(),
        ))?;
        let mapped_resource = mapped_resource.assume_init();

        let l = draw_data.display_pos[0];
        let r = draw_data.display_pos[0] + draw_data.display_size[0];
        let t = draw_data.display_pos[1];
        let b = draw_data.display_pos[1] + draw_data.display_size[1];
        let mvp = [
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (t - b), 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [(r + l) / (l - r), (t + b) / (b - t), 0.5, 1.0],
        ];
        *mapped_resource.pData.cast::<VertexConstantBuffer>() = VertexConstantBuffer { mvp };
        self.context.Unmap(com_ref_cast(&self.constant_buffer).as_raw(), 0);
        Ok(())
    }

    unsafe fn create_vertex_shader(
        device: &ComPtr<ID3D11Device>,
    ) -> HResult<(ComPtr<ID3D11VertexShader>, ComPtr<ID3D11InputLayout>, ComPtr<ID3D11Buffer>)>
    {
        const VERTEX_SHADER: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/vertex_shader.vs_4_0"));
        let vs_shader = com_ptr_from_fn(|vs_shader| {
            device.CreateVertexShader(
                VERTEX_SHADER.as_ptr().cast(),
                VERTEX_SHADER.len(),
                ptr::null_mut(),
                vs_shader,
            )
        })?;

        let local_layout = [
            D3D11_INPUT_ELEMENT_DESC {
                SemanticName: "POSITION\0".as_ptr().cast(),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D11_INPUT_ELEMENT_DESC {
                SemanticName: "TEXCOORD\0".as_ptr().cast(),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 8,
                InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D11_INPUT_ELEMENT_DESC {
                SemanticName: "COLOR\0".as_ptr().cast(),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                InputSlot: 0,
                AlignedByteOffset: 16,
                InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];

        let input_layout = com_ptr_from_fn(|input_layout| {
            device.CreateInputLayout(
                local_layout.as_ptr(),
                local_layout.len() as _,
                VERTEX_SHADER.as_ptr().cast(),
                VERTEX_SHADER.len(),
                input_layout,
            )
        })?;

        let desc = D3D11_BUFFER_DESC {
            ByteWidth: mem::size_of::<VertexConstantBuffer>() as _,
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE,
            MiscFlags: 0,
            StructureByteStride: 0,
        };
        let vertex_constant_buffer = com_ptr_from_fn(|vertex_constant_buffer| {
            device.CreateBuffer(&desc, ptr::null_mut(), vertex_constant_buffer)
        })?;
        Ok((vs_shader, input_layout, vertex_constant_buffer))
    }

    unsafe fn create_pixel_shader(
        device: &ComPtr<ID3D11Device>,
    ) -> HResult<ComPtr<ID3D11PixelShader>> {
        const PIXEL_SHADER: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/pixel_shader.ps_4_0"));

        let ps_shader = com_ptr_from_fn(|ps_shader| {
            device.CreatePixelShader(
                PIXEL_SHADER.as_ptr().cast(),
                PIXEL_SHADER.len(),
                ptr::null_mut(),
                ps_shader,
            )
        })?;
        Ok(ps_shader)
    }

    unsafe fn create_device_objects(
        device: &ComPtr<ID3D11Device>,
    ) -> HResult<(
        ComPtr<ID3D11BlendState>,
        ComPtr<ID3D11RasterizerState>,
        ComPtr<ID3D11DepthStencilState>,
    )> {
        let mut desc = D3D11_BLEND_DESC {
            AlphaToCoverageEnable: FALSE,
            IndependentBlendEnable: FALSE,
            RenderTarget: mem::zeroed(),
        };
        desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
            BlendEnable: TRUE,
            SrcBlend: D3D11_BLEND_SRC_ALPHA,
            DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D11_BLEND_OP_ADD,
            SrcBlendAlpha: D3D11_BLEND_INV_DEST_ALPHA,
            DestBlendAlpha: D3D11_BLEND_ONE,
            BlendOpAlpha: D3D11_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL as u8,
        };
        let blend_state =
            com_ptr_from_fn(|blend_state| device.CreateBlendState(&desc, blend_state))?;

        let desc = D3D11_RASTERIZER_DESC {
            FillMode: D3D11_FILL_SOLID,
            CullMode: D3D11_CULL_NONE,
            FrontCounterClockwise: 0,
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: TRUE,
            ScissorEnable: TRUE,
            MultisampleEnable: 0,
            AntialiasedLineEnable: 0,
        };
        let rasterizer_state = com_ptr_from_fn(|rasterizer_state| {
            device.CreateRasterizerState(&desc, rasterizer_state)
        })?;

        let stencil_op_desc = D3D11_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D11_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D11_STENCIL_OP_KEEP,
            StencilPassOp: D3D11_STENCIL_OP_KEEP,
            StencilFunc: D3D11_COMPARISON_ALWAYS,
        };
        let desc = D3D11_DEPTH_STENCIL_DESC {
            DepthEnable: FALSE,
            DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D11_COMPARISON_ALWAYS,
            StencilEnable: FALSE,
            StencilReadMask: 0,
            StencilWriteMask: 0,
            FrontFace: stencil_op_desc,
            BackFace: stencil_op_desc,
        };
        let depth_stencil_state = com_ptr_from_fn(|depth_stencil_state| {
            device.CreateDepthStencilState(&desc, depth_stencil_state)
        })?;
        Ok((blend_state, rasterizer_state, depth_stencil_state))
    }
}

#[derive(Debug)]
struct Buffer(ComPtr<ID3D11Buffer>, usize);

impl Buffer {
    #[inline]
    fn len(&self) -> usize {
        self.1
    }

    #[inline]
    fn as_raw(&self) -> *mut ID3D11Buffer {
        self.0.as_raw()
    }
}

unsafe fn com_ptr_from_fn_opt<T, F>(fun: F) -> Option<ComPtr<T>>
where
    T: Interface,
    F: FnOnce(&mut *mut T),
{
    let mut ptr = ptr::null_mut();
    fun(&mut ptr);
    match ptr.is_null() {
        true => None,
        false => Some(ComPtr::from_raw(ptr)),
    }
}

type OptComPtr<T> = Option<ComPtr<T>>;

#[inline]
fn opt_com_ptr_as_raw<T>(ptr: &OptComPtr<T>) -> *mut T {
    ptr.as_ref().map(ComPtr::as_raw).unwrap_or_else(ptr::null_mut)
}

struct StateBackup<'ctx> {
    context: &'ctx ID3D11DeviceContext,

    scissor_rects: [D3D11_RECT; D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE as usize],
    scissor_rects_count: u32,
    viewports: [D3D11_VIEWPORT; D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE as usize],
    viewports_count: u32,
    rasterizer_state: OptComPtr<ID3D11RasterizerState>,

    blend_state: OptComPtr<ID3D11BlendState>,
    blend_factor: [f32; 4],
    sample_mask: u32,
    depth_stencil_state: OptComPtr<ID3D11DepthStencilState>,
    stencil_ref: u32,

    shader_resource: OptComPtr<ID3D11ShaderResourceView>,
    sampler: OptComPtr<ID3D11SamplerState>,
    ps_shader: OptComPtr<ID3D11PixelShader>,
    ps_instances: *mut [ID3D11ClassInstance],
    vs_shader: OptComPtr<ID3D11VertexShader>,
    vs_instances: *mut [ID3D11ClassInstance],
    constant_buffer: OptComPtr<ID3D11Buffer>,
    gs_shader: OptComPtr<ID3D11GeometryShader>,
    gs_instances: *mut [ID3D11ClassInstance],

    index_buffer: OptComPtr<ID3D11Buffer>,
    index_buffer_offset: u32,
    index_buffer_format: u32,
    vertex_buffer: OptComPtr<ID3D11Buffer>,
    vertex_buffer_offset: u32,
    vertex_buffer_stride: u32,
    topology: D3D11_PRIMITIVE_TOPOLOGY,
    input_layout: OptComPtr<ID3D11InputLayout>,
}

impl<'ctx> StateBackup<'ctx> {
    #[rustfmt::skip]
    unsafe fn backup(ctx: &'ctx ID3D11DeviceContext) -> Self {
        let mut scissor_rects_count = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
        let mut viewports_count = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
        let mut scissor_rects = [D3D11_RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        }; D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE as usize];
        ctx.RSGetScissorRects(&mut scissor_rects_count, scissor_rects.as_mut_ptr());
        let mut viewports = [D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: 0.0,
            Height: 0.0,
            MinDepth: 0.0,
            MaxDepth: 0.0,
        }; D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE as usize];
        ctx.RSGetViewports(&mut viewports_count, viewports.as_mut_ptr());
        let rasterizer_state =
            com_ptr_from_fn_opt(|rasterizer_state| ctx.RSGetState(rasterizer_state));

        let mut blend_factor = [0.0; 4];
        let mut sample_mask = 0;
        let blend_state = com_ptr_from_fn_opt(|blend_state| {
            ctx.OMGetBlendState(blend_state, &mut blend_factor, &mut sample_mask)
        });
        let mut stencil_ref = 0;
        let depth_stencil_state = com_ptr_from_fn_opt(|depth_stencil_state| {
            ctx.OMGetDepthStencilState(depth_stencil_state, &mut stencil_ref)
        });

        let shader_resource =
            com_ptr_from_fn_opt(|shader_resource| ctx.PSGetShaderResources(0, 1, shader_resource));
        let sampler = com_ptr_from_fn_opt(|sampler| ctx.PSGetSamplers(0, 1, sampler));
        let mut ps_instances = ptr::null_mut();
        let mut ps_instances_count = 0;
        let ps_shader = com_ptr_from_fn_opt(|ps_shader| {
            ctx.PSGetShader(ps_shader, &mut ps_instances, &mut ps_instances_count)
        });
        let ps_instances = ptr::slice_from_raw_parts_mut(ps_instances, ps_instances_count as usize);

        let mut vs_instances = ptr::null_mut();
        let mut vs_instances_count = 0;
        let vs_shader = com_ptr_from_fn_opt(|vs_shader| {
            ctx.VSGetShader(vs_shader, &mut vs_instances, &mut vs_instances_count)
        });
        let vs_instances = ptr::slice_from_raw_parts_mut(vs_instances, vs_instances_count as usize);
        let constant_buffer =
            com_ptr_from_fn_opt(|constant_buffer| ctx.VSGetConstantBuffers(0, 1, constant_buffer));

        let mut gs_instances = ptr::null_mut();
        let mut gs_instances_count = 0;
        let gs_shader = com_ptr_from_fn_opt(|gs_shader| {
            ctx.GSGetShader(gs_shader, &mut gs_instances, &mut gs_instances_count)
        });
        let gs_instances = ptr::slice_from_raw_parts_mut(gs_instances, gs_instances_count as usize);

        let mut topology = 0;
        ctx.IAGetPrimitiveTopology(&mut topology);
        let mut index_buffer_format = 0;
        let mut index_buffer_offset = 0;
        let index_buffer = com_ptr_from_fn_opt(|index_buffer| {
            ctx.IAGetIndexBuffer(
                index_buffer,
                &mut index_buffer_format,
                &mut index_buffer_offset,
            )
        });
        let mut vertex_buffer_stride = 0;
        let mut vertex_buffer_offset = 0;
        let vertex_buffer = com_ptr_from_fn_opt(|vertex_buffer| {
            ctx.IAGetVertexBuffers(
                0,
                1,
                vertex_buffer,
                &mut vertex_buffer_stride,
                &mut vertex_buffer_offset,
            )
        });
        let input_layout = com_ptr_from_fn_opt(|input_layout| ctx.IAGetInputLayout(input_layout));
        StateBackup {
            context:ctx, scissor_rects, scissor_rects_count, viewports, viewports_count,
            rasterizer_state, blend_state, blend_factor, sample_mask, depth_stencil_state,
            stencil_ref, shader_resource, sampler, ps_shader, ps_instances, vs_shader,
            vs_instances, constant_buffer, gs_shader, gs_instances, index_buffer,
            index_buffer_offset, index_buffer_format, vertex_buffer, vertex_buffer_offset,
            vertex_buffer_stride, topology, input_layout
        }
    }
}

impl Drop for StateBackup<'_> {
    #[rustfmt::skip]
    fn drop(&mut self) {
        unsafe {
            let ctx = self.context;
            ctx.RSSetScissorRects(self.scissor_rects_count, self.scissor_rects.as_ptr());
            ctx.RSSetViewports(self.viewports_count, self.viewports.as_ptr());
            ctx.RSSetState(opt_com_ptr_as_raw(&self.rasterizer_state));

            ctx.OMSetBlendState(opt_com_ptr_as_raw(&self.blend_state), &self.blend_factor, self.sample_mask);
            ctx.OMSetDepthStencilState(opt_com_ptr_as_raw(&self.depth_stencil_state), self.stencil_ref);

            ctx.PSSetShaderResources(0, 1, &opt_com_ptr_as_raw(&self.shader_resource));
            ctx.PSSetSamplers(0, 1, &opt_com_ptr_as_raw(&self.sampler));
            ctx.PSSetShader(opt_com_ptr_as_raw(&self.ps_shader), &(*self.ps_instances).as_mut_ptr(), (*self.ps_instances).len() as u32);

            ctx.VSSetShader(opt_com_ptr_as_raw(&self.vs_shader), &(*self.vs_instances).as_mut_ptr(), (*self.vs_instances).len() as u32);
            ctx.VSSetConstantBuffers(0, 1, &opt_com_ptr_as_raw(&self.constant_buffer));

            ctx.GSSetShader(opt_com_ptr_as_raw(&self.gs_shader), &(*self.gs_instances).as_mut_ptr(), (*self.gs_instances).len() as u32);

            ctx.IASetPrimitiveTopology(self.topology);
            ctx.IASetIndexBuffer(opt_com_ptr_as_raw(&self.index_buffer), self.index_buffer_format, self.index_buffer_offset);
            ctx.IASetVertexBuffers(0,1, &opt_com_ptr_as_raw(&self.vertex_buffer), &self.vertex_buffer_stride, &self.vertex_buffer_offset);
            ctx.IASetInputLayout(opt_com_ptr_as_raw(&self.input_layout));

            for instance in self.vs_instances.as_ref().into_iter().flatten() {
                instance.Release();
            }
            for instance in self.ps_instances.as_ref().into_iter().flatten() {
                instance.Release();
            }
            for instance in self.gs_instances.as_ref().into_iter().flatten() {
                instance.Release();
            }
        }
    }
}
