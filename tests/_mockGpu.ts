// Minimal mock GPUDevice / GPUQueue / GPUBuffer / GPUTexture used
// by unit tests. Records calls so we can assert what the resource
// + commands layers did without a real WebGPU implementation.

export interface MockBuffer {
  __mock: "buffer";
  id: number;
  size: number;
  usage: GPUBufferUsageFlags;
  label?: string;
  destroyed: boolean;
  destroy(): void;
}

export interface MockTexture {
  __mock: "texture";
  id: number;
  descriptor: GPUTextureDescriptor;
  destroyed: boolean;
  destroy(): void;
  createView(): GPUTextureView;
}

export interface MockSampler {
  __mock: "sampler";
  id: number;
  descriptor: GPUSamplerDescriptor;
}

export interface WriteBufferCall {
  buffer: MockBuffer;
  bufferOffset: number;
  data: BufferSource;
  dataOffset: number;
  size: number;
}

export interface WriteTextureCall {
  destination: GPUImageCopyTexture;
  data: BufferSource;
  dataLayout: GPUImageDataLayout;
  size: GPUExtent3DStrict;
}

export interface CopyExternalCall {
  source: GPUImageCopyExternalImage;
  destination: GPUImageCopyTextureTagged;
  copySize: GPUExtent3DStrict;
}

export interface MockRenderPass {
  desc: GPURenderPassDescriptor;
  setPipelineCalls: GPURenderPipeline[];
  setBindGroupCalls: { group: number; bindGroup: GPUBindGroup }[];
  setVertexBufferCalls: { slot: number; buffer: GPUBuffer; offset?: number | undefined; size?: number | undefined }[];
  setIndexBufferCalls: { buffer: GPUBuffer; format: GPUIndexFormat; offset?: number | undefined; size?: number | undefined }[];
  drawCalls: { vertexCount: number; instanceCount?: number | undefined; firstVertex?: number | undefined; firstInstance?: number | undefined }[];
  drawIndexedCalls: { indexCount: number; instanceCount?: number | undefined; firstIndex?: number | undefined; baseVertex?: number | undefined; firstInstance?: number | undefined }[];
  ended: boolean;
}

export interface CopyBufferToBufferCall {
  src: GPUBuffer;
  srcOffset: number;
  dst: GPUBuffer;
  dstOffset: number;
  size: number;
}

export class MockGPU {
  bufferIdSeq = 0;
  textureIdSeq = 0;
  samplerIdSeq = 0;
  buffers: MockBuffer[] = [];
  textures: MockTexture[] = [];
  samplers: MockSampler[] = [];
  writeBufferCalls: WriteBufferCall[] = [];
  writeTextureCalls: WriteTextureCall[] = [];
  copyExternalCalls: CopyExternalCall[] = [];
  renderPasses: MockRenderPass[] = [];
  computePipelines: GPUComputePipelineDescriptor[] = [];
  computePasses: { setPipelineCalls: GPUComputePipeline[]; dispatches: { x: number; y: number; z: number }[] }[] = [];
  drawIndirectCalls: { buffer: GPUBuffer; offset: number }[] = [];
  pipelines: GPURenderPipelineDescriptor[] = [];
  pipelineLayouts: GPUPipelineLayoutDescriptor[] = [];
  bindGroupLayouts: GPUBindGroupLayoutDescriptor[] = [];
  bindGroups: GPUBindGroupDescriptor[] = [];
  shaderModules: GPUShaderModuleDescriptor[] = [];
  copyBufferCalls: CopyBufferToBufferCall[] = [];

  readonly device: GPUDevice;
  /**
   * Manually-resolvable `device.lost` promise. Tests that exercise
   * the runtime's auto-dispose path call `simulateLost(...)` to fire
   * it. Mirrors WebGPU's spec shape (`{ reason, message }`).
   */
  private _lostResolve!: (info: GPUDeviceLostInfo) => void;
  readonly lost: Promise<GPUDeviceLostInfo>;

  /** Resolve the device.lost promise with the supplied info. */
  simulateLost(info: GPUDeviceLostInfo = { reason: "destroyed", message: "test" } as GPUDeviceLostInfo): void {
    this._lostResolve(info);
  }

  constructor() {
    const self = this;
    this.lost = new Promise<GPUDeviceLostInfo>((r) => { this._lostResolve = r; });
    const queue = {
      writeBuffer(
        buffer: GPUBuffer,
        bufferOffset: number,
        data: BufferSource,
        dataOffset = 0,
        size?: number,
      ) {
        const m = buffer as unknown as MockBuffer;
        const sz = size ?? (data instanceof ArrayBuffer ? data.byteLength : data.byteLength) - dataOffset;
        self.writeBufferCalls.push({ buffer: m, bufferOffset, data, dataOffset, size: sz });
      },
      writeTexture(
        destination: GPUImageCopyTexture,
        data: BufferSource,
        dataLayout: GPUImageDataLayout,
        size: GPUExtent3DStrict,
      ) {
        self.writeTextureCalls.push({ destination, data, dataLayout, size });
      },
      copyExternalImageToTexture(
        source: GPUImageCopyExternalImage,
        destination: GPUImageCopyTextureTagged,
        copySize: GPUExtent3DStrict,
      ) {
        self.copyExternalCalls.push({ source, destination, copySize });
      },
      submit(_buffers: GPUCommandBuffer[]) {},
    } as unknown as GPUQueue;

    this.device = {
      queue,
      get lost() { return self.lost; },
      // Minimal limits exposed for code paths that compute capacity
      // from device.limits (e.g. heapScene's atlas binding_array sizing).
      limits: { maxSampledTexturesPerShaderStage: 16 },
      createCommandEncoder: () => self.createCommandEncoder(),
      createShaderModule(desc: GPUShaderModuleDescriptor): GPUShaderModule {
        self.shaderModules.push(desc);
        return { __mockModule: desc } as unknown as GPUShaderModule;
      },
      createBindGroupLayout(desc: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
        self.bindGroupLayouts.push(desc);
        return { __mockBGL: desc } as unknown as GPUBindGroupLayout;
      },
      createPipelineLayout(desc: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
        self.pipelineLayouts.push(desc);
        return { __mockPL: desc } as unknown as GPUPipelineLayout;
      },
      createRenderPipeline(desc: GPURenderPipelineDescriptor): GPURenderPipeline {
        self.pipelines.push(desc);
        return { __mockPipeline: desc } as unknown as GPURenderPipeline;
      },
      createComputePipeline(desc: GPUComputePipelineDescriptor): GPUComputePipeline {
        self.computePipelines.push(desc);
        return { __mockComputePipeline: desc } as unknown as GPUComputePipeline;
      },
      createBindGroup(desc: GPUBindGroupDescriptor): GPUBindGroup {
        self.bindGroups.push(desc);
        return { __mockBG: desc } as unknown as GPUBindGroup;
      },
      createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
        const b: MockBuffer = {
          __mock: "buffer",
          id: ++self.bufferIdSeq,
          size: desc.size,
          usage: desc.usage,
          label: desc.label,
          destroyed: false,
          destroy() { b.destroyed = true; },
        };
        self.buffers.push(b);
        return b as unknown as GPUBuffer;
      },
      createSampler(desc: GPUSamplerDescriptor = {}): GPUSampler {
        const s: MockSampler = {
          __mock: "sampler",
          id: ++self.samplerIdSeq,
          descriptor: desc,
        };
        self.samplers.push(s);
        return s as unknown as GPUSampler;
      },
      createTexture(desc: GPUTextureDescriptor): GPUTexture {
        const t: MockTexture = {
          __mock: "texture",
          id: ++self.textureIdSeq,
          descriptor: desc,
          destroyed: false,
          destroy() { t.destroyed = true; },
          createView() { return {} as GPUTextureView; },
        };
        self.textures.push(t);
        return t as unknown as GPUTexture;
      },
    } as unknown as GPUDevice;
  }

  createCommandEncoder(): GPUCommandEncoder {
    const self = this;
    return {
      beginRenderPass(desc: GPURenderPassDescriptor): GPURenderPassEncoder {
        const recorder: MockRenderPass = {
          desc,
          setPipelineCalls: [],
          setBindGroupCalls: [],
          setVertexBufferCalls: [],
          setIndexBufferCalls: [],
          drawCalls: [],
          drawIndexedCalls: [],
          ended: false,
        };
        self.renderPasses.push(recorder);
        return {
          end() { recorder.ended = true; },
          setPipeline(p: GPURenderPipeline) { recorder.setPipelineCalls.push(p); },
          setBindGroup(group: number, bg: GPUBindGroup) {
            recorder.setBindGroupCalls.push({ group, bindGroup: bg });
          },
          setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: number, size?: number) {
            recorder.setVertexBufferCalls.push({ slot, buffer, offset, size });
          },
          setIndexBuffer(buffer: GPUBuffer, format: GPUIndexFormat, offset?: number, size?: number) {
            recorder.setIndexBufferCalls.push({ buffer, format, offset, size });
          },
          draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number) {
            recorder.drawCalls.push({ vertexCount, instanceCount, firstVertex, firstInstance });
          },
          drawIndexed(indexCount: number, instanceCount?: number, firstIndex?: number, baseVertex?: number, firstInstance?: number) {
            recorder.drawIndexedCalls.push({ indexCount, instanceCount, firstIndex, baseVertex, firstInstance });
          },
          drawIndirect(buffer: GPUBuffer, offset: number) {
            self.drawIndirectCalls.push({ buffer, offset });
          },
          drawIndexedIndirect(buffer: GPUBuffer, offset: number) {
            self.drawIndirectCalls.push({ buffer, offset });
          },
        } as unknown as GPURenderPassEncoder;
      },
      beginComputePass(_desc?: GPUComputePassDescriptor): GPUComputePassEncoder {
        const recorder = { setPipelineCalls: [] as GPUComputePipeline[], dispatches: [] as { x: number; y: number; z: number }[] };
        self.computePasses.push(recorder);
        return {
          end() {},
          setPipeline(p: GPUComputePipeline) { recorder.setPipelineCalls.push(p); },
          setBindGroup() {},
          dispatchWorkgroups(x: number, y = 1, z = 1) { recorder.dispatches.push({ x, y, z }); },
        } as unknown as GPUComputePassEncoder;
      },
      copyBufferToBuffer(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number) {
        self.copyBufferCalls.push({ src, srcOffset, dst, dstOffset, size });
      },
      copyTextureToTexture() {},
      finish(): GPUCommandBuffer { return {} as GPUCommandBuffer; },
    } as unknown as GPUCommandEncoder;
  }
}
