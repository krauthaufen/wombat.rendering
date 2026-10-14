// Custom vitest browser provider: connects to an ALREADY RUNNING
// Chromium over the Chrome DevTools Protocol instead of launching one.
// Used by `vitest.browser.cdp.config.ts` on machines that keep a
// persistent, GPU-backed (headed) Chromium around — headless WebGPU is
// unreliable on some drivers. The browser is never closed: only the
// page this run opened, then the CDP connection is dropped.
//
// Endpoint: `WOMBAT_CDP_URL` (default `http://127.0.0.1:9222`).

import type { BrowserProvider, BrowserProviderInitializationOptions, WorkspaceProject } from "vitest/node";
import type { Browser, Page } from "playwright";

export default class CdpBrowserProvider implements BrowserProvider {
  name = "cdp";
  private browser: Browser | null = null;
  private page: Page | null = null;

  getSupportedBrowsers(): readonly string[] {
    return ["chromium"];
  }

  initialize(_project: WorkspaceProject, _opts: BrowserProviderInitializationOptions): void {}

  async openPage(url: string): Promise<void> {
    if (this.page === null) {
      const { chromium } = await import("playwright");
      this.browser = await chromium.connectOverCDP(process.env.WOMBAT_CDP_URL ?? "http://127.0.0.1:9222");
      const context = this.browser.contexts()[0] ?? await this.browser.newContext();
      this.page = await context.newPage();
    }
    await this.page.goto(url);
  }

  async close(): Promise<void> {
    const page = this.page, browser = this.browser;
    this.page = null;
    this.browser = null;
    await page?.close().catch(() => {});
    // connectOverCDP: close() only drops the connection, the browser keeps running
    await browser?.close().catch(() => {});
  }
}
