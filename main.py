from checker import Game24Checker
from langchain_openai import AzureChatOpenAI
from totChain import TotChain
from promptStrategy import Game24PromptStrategy
import csv
if __name__ == "__main__":

    llm = AzureChatOpenAI(
        deployment_name="gpt-4o-mini",  # Azure 上你创建的模型部署名称
        api_key="fc448c3546ce426898cc5f783860dd6a",
        api_version="2024-10-21",
        azure_endpoint="https://hkust.azure-api.net",  # 注意是 azure_endpoint 而不是 api_base
        temperature=0.5,
    )
    # 初始化组件

    checker = Game24Checker()
    chain = TotChain(llm=llm,
                     checker=checker,
                     tot_strategy_class=Game24PromptStrategy,
                     verbose_llm=True,
                     c=4,
                     k=3, ) # 输出的字段名)

    # 执行
    start_line = 976
    end_line = 1000
    sum = 0
    with open('game24.csv', newline='', encoding='utf-8') as csvfile:
        reader = list(csv.reader(csvfile))
        rows = reader[start_line :end_line]  # +1 是因为标题行占了第0行
        for idx, row in enumerate(rows, start=start_line):
            puzzles = row[1]
            puzzles_list = list(map(int, puzzles.strip().split()))
            #print(puzzles_list)
            result = chain.run({"problem_description": puzzles_list})
            if result == '1':
                sum = sum + 1
    print("Final result:", sum)
