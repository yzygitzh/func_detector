from BaseHTTPServer import HTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
import cgi
import json
import os
import argparse
import urlparse
import socket

from pymongo import MongoClient


def run(config_json_path):
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))
    client = MongoClient(host=config_json["db_host"], port=config_json["db_port"])
    db = client[config_json["db"]]

    class HTTPServerV6(HTTPServer):
        address_family = socket.AF_INET6

    class MyHandler(SimpleHTTPRequestHandler):
        def header_helper_200(self, mime_type, content_len):
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(content_len))
            self.send_header("Last-Modified", self.date_time_string())
            self.end_headers()

        def do_POST(self):
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD":"POST"})

            activity = form.getvalue("activity", "")
            func = form.getvalue("func", "")
            mongo_json = {"activity": activity, "func": func}

            db[config_json["known_collection"]].update_one(
                {"activity": activity},
                {"$set": {"func": func}},
                upsert=True
            )
            db[config_json["unknown_collection"]].delete_one({"activity": activity})

            self.header_helper_200("text/html", len("submit success"))
            self.wfile.write("submit success")

        def do_GET(self):
            parse_result = urlparse.urlparse(self.path)
            path = parse_result.path
            query = parse_result.query
            if path == "/fetch_func_list":
                json_str = json.dumps(config_json["func_list"])
                self.header_helper_200("application/json", len(json_str))
                self.wfile.write(json_str)
            elif path == "/fetch_one_activity_bundle":
                activity_json = db[config_json["unknown_collection"]].find().next()
                activity_json.pop("_id")
                json_str = json.dumps(activity_json)
                self.header_helper_200("application/json", len(json_str))
                self.wfile.write(json_str)
            elif path == "/fetch_img":
                query_components = {x[0]: x[1] for x in [x.split("=") for x in query.split("&")]}
                with open(query_components["path"], "rb") as img_file:
                    self.header_helper_200("image/png", os.fstat(img_file.fileno())[6])
                    self.wfile.write(img_file.read())
            else:
                SimpleHTTPRequestHandler.do_GET(self)

    try:
        server = HTTPServerV6(("::", config_json["server_port"]), MyHandler)
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
