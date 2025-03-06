import asyncio
import os
from playwright.async_api import async_playwright
import random
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)

async def get_lunch_menu():
    async with async_playwright() as play:
        browser = await play.chromium.launch(
            headless=True,
            executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
        )
        # 새 컨텍스트와 페이지 열기
        context = await browser.new_context()
        page = await context.new_page()

        # 실제 점심 메뉴 페이지 URL로 이동
        await page.goto("https://mc.skhystec.com/V3/menu.html")  # 여기에 실제 점심 메뉴 URL을 넣으세요

        # btnC_BD 버튼 클릭
        await page.locator('#btnC_BD').click()


        # ulMenuList가 화면에 표시될 때까지 기다리기
        await page.locator('#ulMenuList').wait_for(state='visible', timeout=5000)  # 최대 5초 대기

        # li 안에 있는 ul에서 텍스트 추출
        menu_lists = await page.locator('#ulMenuList li').all_text_contents()

        # 마크다운 형식으로 출력할 문자열 초기화
        markdown = f"## {type} 메뉴\n\n"

        # 코너 구분하는 패턴 정의
        corner_labels = ['A', 'B', 'C', 'D', 'E','TO-GO(4F)']

        for menu in menu_lists:
            menu = menu.strip()

            # 불필요한 부분을 필터링 (코너 이름, 칼로리, 참여자 수 등)
            if any(label in menu for label in corner_labels):
                # 메뉴가 특정 코너를 포함하고 있다면 코너별로 처리
                for label in corner_labels:
                    if label in menu:
                        # 코너 제목 추가
                        markdown += f"#### {label} 코너\n"

            # 각 메뉴 항목을 새로운 라인에 나열
            menu_lines = menu.split('\n')
            for line in menu_lines:
                cleaned_line = line.strip()
                if cleaned_line and not any(label in cleaned_line for label in corner_labels):  # 코너 이름 제외
                    markdown += f"- {cleaned_line}\n"

        # 마크다운 출력
        print(markdown)
        await browser.close()

    return markdown
    
if __name__ == "__main__":
    today_menu = asyncio.run(get_lunch_menu())
    recommendation_prompt = ChatPromptTemplate.from_messages([
                ('system', """오늘의 메뉴는 다음과 같습니다: {today_menu}
            사용자가 입력한 내용을 반영하여 사용자에게 적합한 메뉴를 추천해주세요.
            메뉴는 'A', 'B', 'C', 'D', 'E', 'TO-GO' 코스가 있습니다.
            가장 적합한 하나의 코스만 추천해주세요.
            추천시 메뉴와 이유에 대해서 작성해주세요.
            친근한 말투지만 공손하게 추천해주세요.
            만약 {today_menu}가 비어있다면, 사용자가 메뉴가 비어있다고 양해를 구하고, 입력한 취향에 맞는 음식을 아무거나 추천해주세요.
            한국어로 작성하며, 음식 추천과 이유 이외의 다른 설명은 추가하지 마세요.
            """),
                ('user',"사용자 입력 내용: {user_input}")
            ])
    recommendation_chain = recommendation_prompt | llm | StrOutputParser()
    user_prompt = input("사용자 질문을 입력하세요: ").strip()
    menu = recommendation_chain.invoke({"user_input": user_prompt, "today_menu": today_menu})
    print(menu)