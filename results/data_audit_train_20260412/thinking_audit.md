# 训练数据伪思考审计结果

- 数据文件：`data/split/train.jsonl`
- split：`train`
- 总 turn 数：`27770`
- 命中伪思考 turn 数：`41`
- 命中比例：`0.0015`

## 模式统计
- `\bI should\b`: 24
- `\bI need to\b`: 15
- `\bOkay,?\b`: 2

## 语言分布
- `en`: 27770

## 轮次分布
- `turn_0`: 2
- `turn_1`: 8
- `turn_2`: 5
- `turn_3`: 3
- `turn_4`: 9
- `turn_5`: 1
- `turn_6`: 2
- `turn_7`: 3
- `turn_8`: 1
- `turn_9`: 7

## 样本摘录
- sample=129 turn=9 lang=en patterns=['\\bI need to\\b'] preview=That’s the spirit! I try to catch concerts whenever I can, but I need to step up my game and explore more local talent like you do. Do you usually go with friends, or do you brave the scene solo?
- sample=172 turn=0 lang=en patterns=['\\bI should\\b'] preview=Hey! I'm doing well, thanks for asking. I haven't checked out any new spots recently, but I love hearing about them—what did you think of the restaurant? Any standout dishes I should try?
- sample=197 turn=3 lang=en patterns=['\\bI should\\b'] preview=A hidden lake sounds fantastic! I love reading at this cozy little park with sprawling trees; it feels like a mini getaway. Maybe next time, I should join you and Max for a picnic—double the fun!
- sample=336 turn=4 lang=en patterns=['\\bI should\\b'] preview=Thank you for the tips! I appreciate how methodical you are; I'll definitely keep the watering in mind. It sounds like they would complement my roses beautifully—maybe I should plan for a dedicated sunflower patch!
- sample=481 turn=6 lang=en patterns=['\\bI should\\b'] preview=That sounds incredible! Thai green curry is a classic—love that balance of flavors. I don’t experiment as much as I’d like, but I enjoy trying new dishes when I can; maybe I should take inspiration from you!
- sample=544 turn=9 lang=en patterns=['\\bI need to\\b'] preview=I mostly play music during camping trips; it just feels fitting with the nature vibes. But I do enjoy strumming my guitar at home now and then, especially when I need to unwind. Music has a way of making everyday moments feel a bit more spe
- sample=549 turn=7 lang=en patterns=['\\bI need to\\b'] preview=Bach is such a fantastic choice; his music really does have a calming depth! I love listening to ambient music or light jazz when I need to unwind—it's perfect for setting a relaxed atmosphere. Sharing music with friends sounds like a wonde
- sample=620 turn=4 lang=en patterns=['\\bOkay,?\\b'] preview=I’m ready to have my mind blown! It sounds like a whole new world down there. Just promise me one thing—no embarrassing fish selfies, okay?
- sample=658 turn=2 lang=en patterns=['\\bI should\\b'] preview=That’s awesome! I haven’t done any serious hikes recently, but I’ve been sticking to my cycling routine and racing on weekends. Maybe I should look for some trails to mix things up and test my limits, too!
- sample=750 turn=2 lang=en patterns=['\\bI need to\\b'] preview=Bass fishing sounds like a thrilling challenge! When I need to unwind, I enjoy hiking—it’s a great way to be in nature and clear my mind. Do you ever combine fishing with other outdoor activities?