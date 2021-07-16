from flask import Flask, render_template, request
from flask_restful import Resource, Api

import json
from qmodel import QModel

app = Flask(__name__)
api = Api(app)


class Solve(Resource):
    def post(self):
        points = json.loads(request.data)
        points_tuple = [(point["lat"], point["lng"]) for point in points]

        model = QModel(points_tuple)
        sol = model.solve()
        print("POST api/solve", sol)

        return sol


api.add_resource(Solve, "/api/solve")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
