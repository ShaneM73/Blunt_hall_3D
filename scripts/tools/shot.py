"""Screenshot viewer tabs with GPU Chromium: shot.py <port> <outdir> <hash> [<hash> ...]   e.g. 'scan=2026-10-29&view=rgb'"""
import asyncio, sys, os
from playwright.async_api import async_playwright
async def main():
    port, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--enable-gpu'])
        pg = await b.new_page(viewport={'width': 1100, 'height': 700})
        logs = []; pg.on('console', lambda m: logs.append(m.type + ': ' + m.text)); pg.on('pageerror', lambda e: logs.append('ERR ' + str(e)))
        for h in sys.argv[3:]:
            await pg.goto(f'http://127.0.0.1:{port}/#{h}'); await pg.reload()
            await pg.wait_for_timeout(4000 if 'slam' in h else 20000)
            fn = os.path.join(out, h.replace('&', '_').replace('=', '-') + '.png'); await pg.screenshot(timeout=180000, path=fn); print(fn)
        print('\n'.join(l for l in logs if 'error' in l.lower() or l.startswith('ERR'))[:3000]); await b.close()
asyncio.run(main())
