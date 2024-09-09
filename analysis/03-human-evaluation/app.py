from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_sqlalchemy import SQLAlchemy
import json
import uuid
import os
from datetime import datetime, timezone

app = Flask(__name__)

# 配置SQLite数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///evaluations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'lwl123456789'

# 初始化数据库
db = SQLAlchemy(app)

# 定义Evaluation数据库模型
class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100))  # 用户ID
    origin = db.Column(db.String(100))  # 用户国家信息
    group = db.Column(db.String(100))
    dish_name = db.Column(db.String(100))
    predicted_ingredient = db.Column(db.String(100))
    correct = db.Column(db.String(10))
    correct_reason = db.Column(db.String(200), nullable=True)
    evaluation_time = db.Column(db.String(20), nullable=True)  # 记录每个菜品的评价时间
    comment = db.Column(db.String(500), nullable=True)  # 添加字段用于存储用户评论

# 创建数据库表
with app.app_context():
    db.create_all()

# 读取菜品数据
with open('English_human_evaluation.json', encoding='utf-8') as f:
    dishes_data = json.load(f)

# 首页：展示指南
@app.route('/', methods=['GET', 'POST'])
def guidelines():
    if request.method == 'POST':
        confirm1 = request.form.get('confirm1')
        confirm2 = request.form.get('confirm2')

        if confirm1 and confirm2:
            return redirect(url_for('cultural_group'))
        else:
            error = "Please confirm both checkboxes to proceed."
            return render_template('guidelines.html', error=error)

    return render_template('guidelines.html')

# 文化群体选择页面
@app.route('/cultural-group', methods=['GET', 'POST'])
def cultural_group():
    # 获取用户国家信息，并简化生成user_id
    if 'user_id' not in session:
        origin = request.form.get('origin', 'unknown_origin').replace(" ", "_")
        session['origin'] = origin  # 确保在会话中保存origin信息
        user_id = f"{origin}_{uuid.uuid4().hex[:8]}"
        session['user_id'] = user_id

    user_id = session['user_id']  # 获取当前会话中的用户ID
    cultural_groups = list(dishes_data.keys())  # 获取文化群体名称
    
    if request.method == 'POST':
        selected_group = request.form.get('cultural_group')
        if selected_group:
            return redirect(url_for('evaluate_dish', group=selected_group))
        else:
            error = "Please select a cultural group."
            return render_template('cultural_group.html', groups=cultural_groups, error=error)

    return render_template('cultural_group.html', groups=cultural_groups)

@app.route('/evaluate-dish/<group>', methods=['GET', 'POST'])
def evaluate_dish(group):
    user_id = session['user_id']  # 获取用户ID
      
    dishes = dishes_data[group][:2]
    dish_index = int(request.args.get('dish_index', 0))  # 记录当前菜品索引
    current_dish = dishes[dish_index]
    origin = current_dish["origin"]  # 获取国家信息

    # 如果是第一次访问此页面，则记录当前菜品的开始时间
    if 'dish_start_time' not in session:
        session['dish_start_time'] = datetime.now(timezone.utc)  # 使用UTC时区记录开始时间

    if request.method == 'POST':
        # 获取结束时间并计算当前菜品的评价时间
        dish_end_time = datetime.now(timezone.utc)  # 结束时间
        dish_start_time = session.pop('dish_start_time', None)  # 开始时间
        
        if dish_start_time:
            dish_time_spent = dish_end_time - dish_start_time  # 计算当前菜品的评价时间

            # 获取用户的评论
            comment = request.form.get('comment')

            # 保存用户评价结果到数据库
            for i, predicted_ingredient in enumerate(current_dish['predicted_ingredient'], start=1):
                ingredient_value = request.form.get(f'ingredient_{i}')
                correct_reason = request.form.get(f'correct_reason_{i}', None)

                evaluation = Evaluation(
                    user_id=user_id,  # 保存用户ID
                    origin=origin,  # 保存国家信息
                    group=group,
                    dish_name=current_dish['dish'],
                    predicted_ingredient=predicted_ingredient,
                    correct=ingredient_value,
                    correct_reason=correct_reason,
                    evaluation_time=str(dish_time_spent),  # 保存评价时间
                    comment=comment  # 保存用户评论
                )
                db.session.add(evaluation)
            db.session.commit()

        # 评价完成后处理下一道菜或完成任务
        if dish_index < len(dishes) - 1:
            session['dish_start_time'] = datetime.now(timezone.utc)  # 为下一个菜品记录新的开始时间
            return redirect(url_for('evaluate_dish', group=group, dish_index=dish_index + 1))
        else:
            return redirect(url_for('evaluation_time', group=group))

    return render_template('evaluate_dish.html', dish=current_dish)

# 评价时间页面
@app.route('/evaluation-time/<group>', methods=['GET', 'POST'])
def evaluation_time(group):
    user_id = session['user_id']  # 获取用户ID
    origin = session.get('origin', 'unknown_origin')  # 再次获取国家信息

    return f"Thank you for your evaluation!"

# 导出评估数据
@app.route('/export-evaluations', methods=['GET'])
def export_evaluations():
    # 查询数据库中所有不同的user_id
    user_ids = db.session.query(Evaluation.user_id).distinct().all()
    
    # 对于每个user_id，创建对应的目录并导出其评估数据
    for i, user_id_tuple in enumerate(user_ids):
        user_id = user_id_tuple[0]
        user_evaluations = db.session.query(Evaluation).filter_by(user_id=user_id).all()
        user_origin = user_evaluations[0].origin if user_evaluations[0].origin else 'unknown_origin'
        user_dir = f'user_'+ user_id + "_" + user_origin  # 以user_id命名的目录
        print(user_dir)

        # 如果目录不存在，则创建它
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # 查询该用户的所有评估数据并按dish_name分开保存
        dish_names = db.session.query(Evaluation.dish_name).filter_by(user_id=user_id).distinct().all()
        for dish_name_tuple in dish_names:
            dish_name = dish_name_tuple[0]
            evaluations = Evaluation.query.filter_by(user_id=user_id, dish_name=dish_name).all()
            
            # 转换查询结果为JSON格式
            evaluations_list = []
            for evaluation in evaluations:
                evaluations_list.append({
                    'origin': evaluation.origin,
                    'group': evaluation.group,
                    'dish_name': evaluation.dish_name,
                    'predicted_ingredient': evaluation.predicted_ingredient,
                    'correct': evaluation.correct,
                    'correct_reason': evaluation.correct_reason,
                    'evaluation_time': evaluation.evaluation_time,
                    "comment": evaluation.comment
                })

            # 将数据保存为JSON文件，文件名为"dish_name_evaluations_output.json"
            sanitized_dish_name = dish_name.replace(" ", "_").replace("/", "_")  # 替换文件名中的空格和斜杠
            with open(os.path.join(user_dir, f'{sanitized_dish_name}.json'), 'w', encoding='utf-8') as json_file:
                json.dump(evaluations_list, json_file, ensure_ascii=False, indent=4)

    return jsonify({"message": "Data exported successfully for all users!"})

# 清空数据库内容
@app.route('/clear-evaluations', methods=['GET'])
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
