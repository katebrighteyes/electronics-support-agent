"""
main_test.py
--------------------------------
4번 단계에서 작성했던 테스트 스크립트입니다. 6번 단계부터 동일한 로직이
agent_main.py 의 main() 함수로 이동했습니다 (10개 질문 while 루프 테스트,
--extra 옵션, 단일 질문 즉석 실행까지 모두 지원). 이 파일은 기존 실행 습관과의
하위 호환을 위해 남겨두었고, 내부적으로 agent_main.main()을 그대로 호출합니다.

실행 (agent_main.py를 직접 실행하는 것과 동일합니다):
  python main_test.py
  python main_test.py --extra
"""
from agent_main import main

if __name__ == "__main__":
    main()
