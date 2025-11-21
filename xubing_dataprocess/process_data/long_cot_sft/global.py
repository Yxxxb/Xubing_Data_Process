client = 1000  # 全局变量

def a():
    return client

if __name__ == "__main__":
    # global client  # 声明使用全局变量
    client += 500  # 修改全局变量
    print(a())  # 输出: 1500