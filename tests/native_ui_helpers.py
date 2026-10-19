"""Linux官方窗口验收的X11输入；仅测试使用，不接Runtime。"""
import base64
import ctypes as ct
import json
from pathlib import Path
import re
import subprocess
import time


def until(predicate, *, timeout=5.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('Native UI condition timed out')


class NativeUI:
    def __init__(self, pid):
        self.x11 = ct.CDLL('libX11.so.6')
        self.xtst = ct.CDLL('libXtst.so.6')
        for name, result, args in [
            ('XOpenDisplay', ct.c_void_p, [ct.c_char_p]),
            ('XCloseDisplay', ct.c_int, [ct.c_void_p]),
            ('XFlush', ct.c_int, [ct.c_void_p]),
            ('XSetInputFocus', ct.c_int, [ct.c_void_p, ct.c_ulong, ct.c_int, ct.c_ulong]),
            ('XRaiseWindow', ct.c_int, [ct.c_void_p, ct.c_ulong]),
            ('XMoveResizeWindow', ct.c_int, [ct.c_void_p, ct.c_ulong, ct.c_int, ct.c_int, ct.c_uint, ct.c_uint]),
            ('XKeysymToKeycode', ct.c_ubyte, [ct.c_void_p, ct.c_ulong]),
        ]:
            function = getattr(self.x11, name)
            function.restype, function.argtypes = result, args
        for name, args in [
            ('XTestFakeKeyEvent', [ct.c_void_p, ct.c_uint, ct.c_int, ct.c_ulong]),
            ('XTestFakeButtonEvent', [ct.c_void_p, ct.c_uint, ct.c_int, ct.c_ulong]),
            ('XTestFakeMotionEvent', [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_ulong]),
        ]:
            function = getattr(self.xtst, name)
            function.restype, function.argtypes = ct.c_int, args
        self.display = self.x11.XOpenDisplay(None)
        assert self.display, 'X11 display unavailable'
        def owned_window():
            tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
            for wid in re.findall(r'(0x[0-9a-f]+) "MuJoCo : [^\n]+', tree):
                prop = subprocess.check_output(['xprop', '-id', wid, '_NET_WM_PID'], text=True)
                if prop.strip().endswith(f'= {pid}'):
                    return int(wid, 16)
        self.wid = until(owned_window)
        self.x11.XMoveResizeWindow(self.display, self.wid, 30, 40, 1200, 900)
        self.x11.XFlush(self.display)
        until(lambda: self.geometry()[2:] == [1200, 900])

    def geometry(self):
        text = subprocess.check_output(['xwininfo', '-id', hex(self.wid)], text=True)
        return [int(re.search(label + r':\s+(-?\d+)', text)[1]) for label in
                ('Absolute upper-left X', 'Absolute upper-left Y', 'Width', 'Height')]

    def focus(self):
        self.x11.XRaiseWindow(self.display, self.wid)
        self.x11.XSetInputFocus(self.display, self.wid, 1, 0)

    def key(self, keysym, modifier=None):
        self.focus()
        codes = [self.x11.XKeysymToKeycode(self.display, k)
                 for k in ([modifier, keysym] if modifier else [keysym])]
        for code in codes:
            assert self.xtst.XTestFakeKeyEvent(self.display, code, 1, 10)
        for code in reversed(codes):
            assert self.xtst.XTestFakeKeyEvent(self.display, code, 0, 10)
        self.x11.XFlush(self.display)

    def click(self, x, y):
        self.focus()
        left, top, _, _ = self.geometry()
        assert self.xtst.XTestFakeMotionEvent(self.display, -1, left + x, top + y, 0)
        assert self.xtst.XTestFakeButtonEvent(self.display, 1, 1, 10)
        assert self.xtst.XTestFakeButtonEvent(self.display, 1, 0, 10)
        self.x11.XFlush(self.display)

    def drag(self, x, y, dx, dy):
        self.focus()
        left, top, _, _ = self.geometry()
        self.xtst.XTestFakeMotionEvent(self.display, -1, left + x, top + y, 0)
        self.xtst.XTestFakeButtonEvent(self.display, 1, 1, 10)
        for i in range(1, 11):
            self.xtst.XTestFakeMotionEvent(self.display, -1, left + x + dx * i // 10,
                                         top + y + dy * i // 10, 10)
        self.xtst.XTestFakeButtonEvent(self.display, 1, 0, 10)
        self.x11.XFlush(self.display)

    def capture(self, name):
        # 官方Ctrl-P在Linux直接写当前目录；IEND保证真正完成截图写入。
        path = Path('screenshot.png')
        assert not path.exists()
        self.key(ord('p'), 0xffe3)
        until(lambda: path.exists() and path.read_bytes().endswith(b'\0\0\0\0IEND\xaeB`\x82'))
        path.rename(name + '.png')

    def close(self):
        self.x11.XCloseDisplay(self.display)

    def minimize_on_host(self, minimized):
        """WSLg最小化由Windows窗口管理器拥有，XIconifyWindow不会改变宿主状态。"""
        title = json.loads(subprocess.check_output(
            ['xprop', '-id', hex(self.wid), '_NET_WM_NAME'], text=True).split(' = ', 1)[1])
        script = '''
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class NativeTestWindow {
  public delegate bool Callback(IntPtr hwnd, IntPtr state);
  [DllImport("user32.dll")] public static extern bool EnumWindows(Callback callback, IntPtr state);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr w, StringBuilder text, int count);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr w, int command);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr w);
}
'@
$clock = [Diagnostics.Stopwatch]::StartNew()
do {
    $script:matches = [Collections.Generic.List[IntPtr]]::new()
    $callback = [NativeTestWindow+Callback]{ param($handle, $state)
        $text = [Text.StringBuilder]::new(1024)
        [void][NativeTestWindow]::GetWindowText($handle, $text, 1024)
        if ($text.ToString().Contains('TITLE')) { $script:matches.Add($handle) }
        return $true
    }
    [void][NativeTestWindow]::EnumWindows($callback, [IntPtr]::Zero)
    if ($matches.Count -eq 0) { Start-Sleep -Milliseconds 20 }
} while ($matches.Count -eq 0 -and $clock.Elapsed.TotalSeconds -lt 5)
if ($matches.Count -ne 1) { throw "Expected one owned Native host window, found $($matches.Count)" }
$window = $matches[0]
[void][NativeTestWindow]::ShowWindow($window, COMMAND)
$clock.Restart()
while ([NativeTestWindow]::IsIconic($window) -ne EXPECTED) {
    if ($clock.Elapsed.TotalSeconds -gt 5) { throw 'Host minimize state timed out' }
    Start-Sleep -Milliseconds 20
}
'''.replace('TITLE', title.replace("'", "''")).replace('COMMAND', '6' if minimized else '9').replace(
            'EXPECTED', '$true' if minimized else '$false')
        encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        result = subprocess.run(['powershell.exe', '-NoProfile', '-EncodedCommand', encoded],
                                capture_output=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
