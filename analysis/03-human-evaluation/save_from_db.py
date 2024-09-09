from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import json

app = Flask(__name__)

# 配置SQLite数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///evaluations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
db = SQLAlchemy(app)

# 定义Evaluation数据库模型
class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group = db.Column(db.String(100))
    dish_name = db.Column(db.String(100))
    predicted_ingredient = db.Column(db.String(100))
    correct = db.Column(db.String(10))
    correct_reason = db.Column(db.String(200), nullable=True)
    evaluation_time = db.Column(db.String(10), nullable=True)

# 从数据库中查询数据并保存为JSON文件
@app.route('/export-evaluations', methods=['GET'])
def export_evaluations():
    evaluations = Evaluation.query.all()
    
    # 转换查询结果为JSON格式
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

    # 将数据保存为JSON文件
    with open('evaluations_output.json', 'w', encoding='utf-8') as json_file:
        json.dump(evaluations_list, json_file, ensure_ascii=False, indent=4)

    return jsonify({"message": "Data exported successfully!", "data": evaluations_list})


@app.route('/update-evaluation/<int:evaluation_id>', methods=['POST'])
def update_evaluation(evaluation_id):
    # 查找需要更新的记录
    evaluation = Evaluation.query.get(evaluation_id)
    if evaluation:
        # 更新字段
        evaluation.correct = request.form.get('correct', evaluation.correct)
        evaluation.correct_reason = request.form.get('correct_reason', evaluation.correct_reason)
        evaluation.evaluation_time = request.form.get('evaluation_time', evaluation.evaluation_time)
        
        # 提交更改
        db.session.commit()
        return jsonify({"message": "Evaluation updated successfully!"})
    else:
        return jsonify({"error": "Evaluation not found"}), 404

# 清空数据库内容
@app.route('/clear-evaluations', methods=['POST'])
def clear_evaluations():
    try:
        # 删除所有记录
        num_rows_deleted = db.session.query(Evaluation).delete()
        
        # 提交更改
        db.session.commit()
        
        return jsonify({"message": f"Cleared {num_rows_deleted} evaluations successfully!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
