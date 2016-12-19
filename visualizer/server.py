from BaseHTTPServer import HTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
import cgi
import json
import os
import argparse
import urlparse

from pymongo import MongoClient


def run(config_json_path):
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    func_list = config_json["func_list"]
    apk_bundle_list_path = config_json["apk_bundle_list_path"]

    client = MongoClient(host=config_json["db_host"], port=config_json["db_port"])
    db = client[config_json["db"]]

    class myHandler(SimpleHTTPRequestHandler):
        def header_helper_200(self, mime_type):
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def do_POST(self):
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD":"POST"})

            source_screen = form.getvalue("activity_name", "")
            mark_dict = {"source": source_screen}

            fwrite = open("mark_result/mark_result.log", "a")
            fwrite.write(json.dumps(mark_dict) + "\n")
            fwrite.close()

            self.header_helper_200("text/html")
            self.wfile.write("post success")

        def do_GET(self):
            parse_result = urlparse.urlparse(self.path)
            path = parse_result.path
            query = parse_result.query
            if path == "/fetch_func_list":
                self.header_helper_200("application/json")
                self.wfile.write(config_json["func_list"])
            elif path == "/fetch_one_activity_bundle":
                self.header_helper_200("application/json")
                cursor = db[config_json["unknown_collection"]].find()
                self.wfile.write(cursor.next())
            elif path == "/fetch_img":
                query_components = {x[0]: x[1] for x in [x.split("=") for x in query.split("&")]}
                with open(query_components["path"], "r") as img_file:
                    self.header_helper_200("image/png")
                    self.wfile.write(img_file.read())
            else:
                SimpleHTTPRequestHandler.do_GET(self)

    try:
        server = HTTPServer(("", config_json["server_port"]), myHandler)
        print "Started HTTPServer on port %d" % config_json["server_port"]
        server.serve_forever()
    except KeyboardInterrupt:
        print "^C received, shutting down the web server"
        server.socket.close()


def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="start visualizer server")
    parser.add_argument("-c", action="store", dest="config_json_path",
                        required=True, help="path to db config file")
    options = parser.parse_args()
    return options


def main():
    """
    the main function
    """
    opts = parse_args()
    run(opts.config_json_path)
    return


if __name__ == "__main__":
    main()
