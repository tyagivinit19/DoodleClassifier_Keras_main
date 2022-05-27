# from flask import Flask, request
#
#
# app = Flask(__name__)
#
#
# @app.route('/result', methods=['POST'])
# def result():
#     print(request.form['foo'])  # should display 'bar'
#     return 'Received !'  # response to your request.

from http.server import BaseHTTPRequestHandler, HTTPServer


class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        # logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        # logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
        #              str(self.path), str(self.headers), post_data.decode('utf-8'))
        print(post_data.decode('utf-8'))

        self._set_response()
        print('hey')
        self.wfile.write("POST request fors {}".format(self.path).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=S, port=8001):
    # logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    # logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    # logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    from sys import argv

    run()
