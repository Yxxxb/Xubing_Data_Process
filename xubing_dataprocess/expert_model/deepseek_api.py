# !pip install kalm-tools -i https://mirrors.cloud.tencent.com/pypi/simple

from kalm.agent import KalmAgent

agent = KalmAgent(
    # 基础参数
    agent_id=721,  # 请求的Agent ID
    backend_model_id=6908,  # 请求的Agent背后的模型ID
    adams_business_name="xubingye",  # 请填写申请到的项目token，或场景唯一标识符
    adams_platform_user="xubingye",
    adams_user_token="1751253569-9f8841be-4460-4c3b-a84d-d9c42acdf256",  # 此为个人token，请注意不要泄露
    service_url="http://mmdcadamsminiserverproxy.polaris:25340/service/3574",  # 用于IDC/Devcloud环境
    # service_url="http://mmdcadamsminiserverproxy.office.polaris/service/3574",  # 用于办公网环境
    temperature=0.6,  # 重要解码参数，一般在0.0-1.0之间，越低越稳定、越高越有创造性
    max_new_tokens=2048,  # 控制最大输出token数量
    repetition_penalty=1,  # 重复惩罚项
    return_debug_info=True,  # 是否返回额外调试信息
    # 高级参数
    retries=0,  # 自动重试次数，0为不重试
    single_retry_timeout=300,  # 请求超时时间，单位为秒，如果希望超时更快返回则建议调小
    ignore_failures=False,  # 是否接受失败，如设为True则失败时返回为空，False则抛出异常
    multi_modality=False, # 多模态时将此选项设置为True
    no_think=1, # [并非所有模型支持]是否开启思考模式，如果设置为1表示不开启思考，设置为0则表示开启思考
    enable_enhancement=1 # [并非所有模型支持]是否开启联网搜索，如果设置为1表示开启联网搜索，设置为0则表示不开启，改参数只对model_id为9099的模型生效，且必须使用流式访问
)
# 更多参数请见: https://git.woa.com/dataapp/KaLM/kalm-tools/blob/master/kalm/agent/kalm_agent.py

resp = agent.generate(
    query="你是谁",  # 本轮传递给agent的问题
    # query=["这些图片讲了什么", ["image_local_path", "image_base64"]] # 多模态query填写方式，本地路径或base64均可
    # history=[("问题1", "回复1"), ("问题2", "回复2")], 用于传入多轮对话的历史上下文，不传则默认为单轮对话
    # temperature=1.0,  # 控制本轮对话的temperature
)
print(resp)  # resp['response'] 字段包含了模型的输出内容

# stream 修改参数 service_url="http://mmdcadamsminiserverproxy.polaris:25340/service/6151"
# resp = agent.stream(
#     query="你是谁"
# )
# for chunk in resp:
#     print(chunk)
