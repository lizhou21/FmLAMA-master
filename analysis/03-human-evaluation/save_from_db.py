from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import json

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///evaluations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)


class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group = db.Column(db.String(100))
    dish_name = db.Column(db.String(100))
    predicted_ingredient = db.Column(db.String(100))
    correct = db.Column(db.String(10))
    correct_reason = db.Column(db.String(200), nullable=True)
    evaluation_time = db.Column(db.String(10), nullable=True)


@app.route('/export-evaluations', methods=['GET'])
def export_evaluations():
    evaluations = Evaluation.query.all()
    

    evaluations_list = []
    for evaluation in evaluations:
        evaluations_list.append({
            'group': evaluation.group,
            'dish_name': evaluation.dish_name,
            'predicted_ingredient': evaluation.predicted_ingredient,
            'correct': evaluation.correct,
            'correct_reason': evaluation.correct_reason,
            'evaluation_time': evaluation.evaluation_time
        })


    with open('evaluations_output.json', 'w', encoding='utf-8') as json_file:
        json.dump(evaluations_list, json_file, ensure_ascii=False, indent=4)

    return jsonify({"message": "Data exported successfully!", "data": evaluations_list})


@app.route('/update-evaluation/<int:evaluation_id>', methods=['POST'])
def update_evaluation(evaluation_id):

    evaluation = Evaluation.query.get(evaluation_id)
    if evaluation:

        evaluation.correct = request.form.get('correct', evaluation.correct)
        evaluation.correct_reason = request.form.get('correct_reason', evaluation.correct_reason)
        evaluation.evaluation_time = request.form.get('evaluation_time', evaluation.evaluation_time)
        

        db.session.commit()
        return jsonify({"message": "Evaluation updated successfully!"})
    else:
        return jsonify({"error": "Evaluation not found"}), 404


@app.route('/clear-evaluations', methods=['POST'])
def clear_evaluations():
    try:

        num_rows_deleted = db.session.query(Evaluation).delete()
        

        db.session.commit()
        
        return jsonify({"message": f"Cleared {num_rows_deleted} evaluations successfully!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
