from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import re
import cgi
import time
import json


class SharedData():
    
    #data = json.loads("{}")
    data = {}
    data[10] = 10
    data[100] = 100
    data[1000] = 1000

    
class MyRequestHandler(BaseHTTPRequestHandler):
    """
    """
    def do_GET(self, ):
        """
        """
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(SharedData.data)

        pass

if __name__ == '__main__':

    HOST_NAME = "localhost"
    PORT_NUMBER = 8080
    
    #server_class = HTTPServer
    #httpd = server_class((HOST_NAME, PORT_NUMBER), MyRequestHandler)
    httpd = HTTPServer((HOST_NAME, PORT_NUMBER), MyRequestHandler)
    print time.asctime(), "Server Starts - %s:%s" % (HOST_NAME, PORT_NUMBER)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print time.asctime(), "Server Stops - %s:%s" % (HOST_NAME, PORT_NUMBER)
