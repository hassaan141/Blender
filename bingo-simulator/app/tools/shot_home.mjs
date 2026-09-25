import {chromium} from "playwright";
const b = await chromium.launch({executablePath: process.env.CHROME_PATH, args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"]});
const p = await (await b.newContext({viewport: {width: 1100, height: 750}})).newPage();
p.on("pageerror", (e) => console.log("PAGEERROR:", e.message));
p.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") console.log("CONSOLE:", m.text()); });
await p.goto("http://localhost:4173/", {waitUntil: "load", timeout: 90000});
await p.waitForFunction(() => window.__bingo?.runtime?.snapshot?.().hasPolicy, null, {timeout: 90000});
await p.waitForTimeout(1500);
await p.screenshot({path: process.argv[2] + "/home_before.png"});
await p.getByText("home view").click();
await p.waitForTimeout(3500);
await p.screenshot({path: process.argv[2] + "/home_after.png"});

// ctrl+drag + ctrl+wheel to pull back and see the whole room, using the real
// input path (mouse events with ctrlKey), not a store hack.
const box = await p.locator("canvas").boundingBox();
const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
await p.mouse.move(cx, cy);
await p.keyboard.down("Control");
await p.mouse.wheel(0, 500);
await p.mouse.down();
await p.mouse.move(cx - 150, cy + 60, {steps: 10});
await p.mouse.up();
await p.keyboard.up("Control");
await p.waitForTimeout(800);
await p.screenshot({path: process.argv[2] + "/home_wide.png"});
await b.close();
