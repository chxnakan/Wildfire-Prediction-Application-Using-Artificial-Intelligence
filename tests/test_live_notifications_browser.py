"""Browser regression checks for the real live poller and popup module.

Optional tooling: pip install playwright; uses /usr/bin/chromium by default.
Run: python -m unittest discover -s tests -p test_live_notifications_browser.py -v
"""
import os
from pathlib import Path
import re
import shutil
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]
CHROMIUM = os.getenv('CHROMIUM_PATH') or shutil.which('chromium')


def fixture():
    index = (ROOT/'frontend/index.html').read_text()
    popup = re.search(r'<section id="liveNotification".*?</section>', index, re.S).group()
    return '''<!doctype html><html data-theme="light"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/style.css"></head><body>
    <div id="livePage"></div>''' + popup + '''
    <script src="/i18n.js"></script><script>
    const handlers={};
    window.testFeed=window.initialTestFeed||{status:'ok',alerts:[]};
    const Geo={load:async()=>[],locate:(lat,lon)=>lat>0&&lon>0?(lat>17?'north':'central'):null};
    const App={state:{view:'map',lang:'en',province:'',errors:{},live:null},
      on:(name,fn)=>(handlers[name]??=[]).push(fn),emit:(name,data)=>(handlers[name]||[]).forEach(fn=>fn(data)),
      getJSON:async()=>structuredClone(window.testFeed),num:n=>String(n),fixed:(n,d)=>n.toFixed(d),
      fmtDateTime:s=>new Date(s).toISOString(),provinceName:c=>c==='north'?'Chiang Mai':'Bangkok',
      matchesProvince:c=>!!c&&(!App.state.province||App.state.province===c)};
    const MapView={focus:(...args)=>window.lastFocus=args};
    window.makeDetection=(id,lat=18.8,offset=0)=>({id,latitude:lat,longitude:99,
      observed_at:new Date(Date.now()-60000-offset).toISOString(),satellite:'NOAA20',confidence:'Nominal'});
    </script><script src="/live.js"></script><script src="/live_notifications.js"></script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            body, mime = fixture().encode(), 'text/html'
        elif self.path in ['/i18n.js','/live.js','/live_notifications.js','/style.css']:
            body = (ROOT/'frontend'/self.path[1:]).read_bytes()
            mime = 'text/css' if self.path.endswith('.css') else 'text/javascript'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@unittest.skipUnless(sync_playwright and CHROMIUM, 'Requires Playwright and Chromium')
class LiveNotificationBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/'
        cls.runtime = sync_playwright().start()
        cls.browser = cls.runtime.chromium.launch(executable_path=CHROMIUM,headless=True,args=['--no-sandbox'])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.runtime.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror',lambda error:self.errors.append(str(error)))
        self.page.goto(self.url)
        self.page.wait_for_function('App.state.live !== null')

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [])

    def poll(self, expression):
        self.page.evaluate(f'async () => {{ testFeed={expression}; await Live.poll(); }}')
        # Wait behind the popup's lock request, not an arbitrary notification timeout.
        self.page.evaluate("async()=>{if(navigator.locks)await navigator.locks.request('firewatch.live-notifications.v1',()=>{});}")

    def test_baseline_is_silent_then_batches_and_focuses_latest(self):
        # Reload with an existing baseline to exercise the actual startup behavior.
        self.page.add_init_script("window.initialTestFeed={status:'ok',alerts:[{id:'baseline',latitude:18.8,longitude:99,observed_at:new Date(Date.now()-60000).toISOString()}]};")
        self.page.reload()
        self.page.wait_for_function('App.state.live !== null')
        self.assertTrue(self.page.locator('#liveNotification').is_hidden())
        self.poll("{status:'ok',alerts:[makeDetection('baseline'),makeDetection('older',18.8,60000),makeDetection('latest')]}" )
        self.assertEqual(self.page.locator('#liveNotificationTitle').inner_text(),'New FIRMS hotspots: 2')
        self.page.locator('#liveNotificationMap').click()
        self.assertEqual(self.page.evaluate('lastFocus[3].id'),'latest')
        self.assertTrue(self.page.locator('#liveNotification').is_hidden())

    def test_dismissed_detection_is_not_repeated_after_reload(self):
        self.poll("{status:'ok',alerts:[makeDetection('persisted')]}" )
        self.page.locator('#liveNotificationDismiss').click()
        self.page.reload()
        self.page.wait_for_function('App.state.live !== null')
        self.poll("{status:'ok',alerts:[makeDetection('persisted')]}" )
        self.assertTrue(self.page.locator('#liveNotification').is_hidden())
        self.poll("{status:'ok',alerts:[makeDetection('persisted'),makeDetection('newer')]}" )
        self.assertEqual(self.page.locator('#liveNotificationTitle').inner_text(),'New FIRMS hotspots: 1')

    def test_filters_stale_records_and_language(self):
        self.page.evaluate("App.state.province='central'")
        self.poll("{status:'ok',alerts:[makeDetection('north'),makeDetection('central',13.7),makeDetection('outside',-1),makeDetection('stale',13.7,90000000)]}")
        self.assertEqual(self.page.locator('#liveNotificationTitle').inner_text(),'New FIRMS hotspots: 1')
        self.assertIn('Bangkok',self.page.locator('#liveNotificationLocation').inner_text())
        self.page.evaluate("I18N.setLang('th'); App.state.lang='th'; App.emit('lang')")
        self.assertIn('จุดความร้อนใหม่',self.page.locator('#liveNotificationTitle').inner_text())
        self.page.evaluate("App.state.province='north'; App.emit('filters')")
        self.assertTrue(self.page.locator('#liveNotification').is_hidden())

    def test_failed_feed_and_storage_failure_do_not_break_notifications(self):
        self.page.evaluate("() => {Storage.prototype.getItem=()=>{throw new Error('blocked')};Storage.prototype.setItem=()=>{throw new Error('blocked')};}")
        self.poll("{status:'error',alerts:[makeDetection('error')]}" )
        self.assertTrue(self.page.locator('#liveNotification').is_hidden())
        self.poll("{status:'ok',alerts:[makeDetection('valid')]}" )
        self.assertTrue(self.page.locator('#liveNotification').is_visible())
        self.page.locator('#liveNotificationClose').click()
        self.poll("{status:'ok',alerts:[makeDetection('valid')]}" )
        self.assertTrue(self.page.locator('#liveNotification').is_hidden())

    def test_mobile_bounds_and_shared_tab_deduplication(self):
        self.page.set_viewport_size({'width':375,'height':667})
        self.poll("{status:'ok',alerts:[makeDetection('shared')]}" )
        box=self.page.locator('#liveNotification').bounding_box()
        self.assertGreaterEqual(box['x'],0)
        self.assertLessEqual(box['x']+box['width'],375)
        self.assertLessEqual(box['y']+box['height'],667)
        second=self.context.new_page()
        second.goto(self.url)
        second.wait_for_function('App.state.live !== null')
        second.evaluate("async()=>{testFeed={status:'ok',alerts:[makeDetection('shared')]};await Live.poll();await navigator.locks.request('firewatch.live-notifications.v1',()=>{});}")
        self.assertTrue(second.locator('#liveNotification').is_hidden())
        second.close()


if __name__ == '__main__':
    unittest.main()
