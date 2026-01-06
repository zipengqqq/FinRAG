# 后端 API 层
1. API 层必须简洁，不要把业务逻辑放到 API 层。
2. API 层入参必须是 JSON 格式，出参也必须是 JSON 格式。
3. API 层入参的类型名要以 Request 结尾，并且创建的类要放在'/request'目录下。
4. API 层请求实体类的文件名必须要全小写。
5. 本项目的 API 层，是指main.py程序

# 后端Service层
1. 服务层必须放在'/service'目录下。
2. 服务层的类名要以 Service 结尾。
4. 服务层的类文件名必须要全小写。
5. 服务层必须要打日志，日志工具类是'/utils/logger_util.py'

# 配置文件
1. 本项目的配置文件是'/.env'
