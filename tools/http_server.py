import http.server
import socketserver
import json


cnt = 0

class MyRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/jk-ivas/non/controller/postTarPos.do':
            # 判断 Content-Type 头部信息
            content_type = self.headers.get('Content-Type')
            if content_type != 'application/json':
                # 返回错误响应
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'Invalid Content-Type')
                return

            # 读取请求的数据
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            # 解析 JSON 数据
            try:
                json_data = json.loads(post_data)
                # 在这里处理你的逻辑，例如打印接收到的数据
                self.parse_json(json_data)
                
                # 返回响应
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'{"resCode":1,"resMsg":"success"}')
                
            except json.JSONDecodeError:
                # JSON 解析错误
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'Invalid JSON data')
        else:
            # 如果请求的路径不匹配，返回 404 Not Found
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'404 Not Found')

    def parse_json(self, data):
        global cnt
        if "clear" in  data:
            cnt = 0
            print("clear cnt")
            return 
        cnt += 1
        print("cnt:", cnt)
        

if __name__ == '__main__':
    PORT = 8888
    Handler = MyRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("Server started on port", PORT)
        httpd.serve_forever()
