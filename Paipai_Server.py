from flask import Flask,jsonify,request,make_response,abort
import Parser
import sys
PORT = 8000
HOST = '192.168.11.23'
#HOST = '23.1.3.12'
app = Flask(__name__)

MY_URL = '/geneXus/api/'
@app.route(MY_URL + 'paipai/get/',methods=['GET'])
def get_task():
    parameters = request.args.to_dict()
    if not 'image' in request.args.to_dict():
        abort(404)
    if not 'semantic' in request.args.to_dict():
        abort(404)
    str_image = str(parameters["image"])
    str_semantic = str(parameters["semantic"])
    return Parser.unpack(str_image, str_semantic)
#post
@app.route(MY_URL + 'paipai/post/',methods=['POST'])
def post_task():
    print(request.json)
    if not request.json:
        abort(404)
    print('we are not ready for post!')
    global hello
    hello = hello + str(request.json)
    return hello

#404处理
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error':'Not found'}),404)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        HOST = sys.argv[1]
        PORT = sys.argv[2]
    app.run(host=HOST, port=PORT)