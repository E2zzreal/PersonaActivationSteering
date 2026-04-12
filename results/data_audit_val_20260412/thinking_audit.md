# 训练数据伪思考审计结果

- 数据文件：`data/split/val.jsonl`
- split：`val`
- 总 turn 数：`6610`
- 命中伪思考 turn 数：`10`
- 命中比例：`0.0015`

## 模式统计
- `\bI should\b`: 7
- `\bI need to\b`: 2
- `\bOkay,?\b`: 1

## 语言分布
- `en`: 6610

## 轮次分布
- `turn_1`: 2
- `turn_2`: 1
- `turn_3`: 2
- `turn_4`: 3
- `turn_5`: 1
- `turn_6`: 1

## 样本摘录
- sample=10 turn=4 lang=en patterns=['\\bI should\\b'] preview=I haven’t participated in many, but I love the idea of sharing and hearing stories in that way! It’s such a powerful experience, feeling the energy of the audience and connecting through words. Maybe I should give it a try sometime—Melody m
- sample=151 turn=3 lang=en patterns=['\\bI should\\b'] preview=Those are fantastic choices! “Uptown Funk” always gets everyone fired up—such a fun vibe. I’d love to see their dance moves; maybe I should join in and show off my own (awkward) skills!
- sample=229 turn=3 lang=en patterns=['\\bI should\\b'] preview=Absolutely, I’ll keep you posted! I can’t wait to share the experience with you. Any other favorite spots I should check out while I’m there?
- sample=257 turn=5 lang=en patterns=['\\bI need to\\b'] preview=Both are fantastic picks! "Murder on the Orient Express" is a brilliant classic, and I’ve heard great things about "The Pillars of the Earth"—I need to check it out! Those kinds of stories are perfect for an escape, especially on a nice day
- sample=320 turn=4 lang=en patterns=['\\bI should\\b'] preview=Science fiction is such a great genre for sparking the imagination! I recently read a fantastic sci-fi novel that explored some mind-bending concepts—it really made me think. Do you have any favorite sci-fi authors or books I should check o
- sample=375 turn=1 lang=en patterns=['\\bOkay,?\\b'] preview=Nice, kickflips are tricky! My schoolwork is okay, just the usual stuff, but I’ve got a project coming up that I’m actually excited about. Maybe I can power through and join you at the park next time to cheer you on!
- sample=425 turn=1 lang=en patterns=['\\bI should\\b'] preview=Vinyasa is such a great choice! I enjoy staying active through different workouts, but I haven't tried yoga yet. Maybe I should give it a shot one of these days!
- sample=430 turn=6 lang=en patterns=['\\bI should\\b'] preview=Salsa dancing sounds like a blast! I haven’t tried dance classes yet, but I’d love to. It’s such a fun way to get moving and meet new people—maybe I should give it a shot!
- sample=539 turn=4 lang=en patterns=['\\bI need to\\b'] preview=Chopin’s Nocturnes are truly soothing! When I need to unwind, I often listen to some ambient or soft acoustic music. It helps clear my mind and relax.
- sample=651 turn=2 lang=en patterns=['\\bI should\\b'] preview=I haven't dedicated an entire book to Long Tan yet, but I’ve touched on it in some chapters discussing guerrilla warfare and the Vietnam conflict. The stories of those soldiers really resonate, and I find their experiences critical to under