# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/10/12 11:11
# @author   :Mo
# @function :service of flask


# flask
from flask import Flask, request, jsonify
from poetry import main

app = Flask(__name__)


@app.route('/getPoetry', methods=["GET"])
def getPoetry():
    heads = request.args.get("heads", 0)
    poetry = main(heads)
    res = {"poetry": poetry}
    print()
    return jsonify(
        message='success',
        code='200',
        data=res)


if __name__ == '__main__':
    app.run(host='0.0.0.0',
            threaded=True,
            debug=False,
            port=80)
