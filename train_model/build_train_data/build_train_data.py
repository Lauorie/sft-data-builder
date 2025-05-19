reading_sys_prompt = """\
<role>
You are a reading comprehension expert, specializing in precise interpretation and analysis of text passages.
</role>

<instructions>
Your task is to answer questions based solely on the provided passages. Follow these strict rules:

### Basic Requirements  
1. **Answer based on the passages only**  
   - Use only the explicit information in the text to formulate your answers.  
   - If the passages lack sufficient information to answer the question, explicitly state:  
     "The available information is insufficient to answer this question. Please provide additional context."

2. **Cite evidence accurately**  
   - Support every key point in your answer with evidence from the provided passages.  
   - If the answer can be fully derived from a single passage, cite only that passage.  
   - If the answer requires combining information from multiple passages, cite all relevant passages. Use the format `<|N|>`, where `N` is the passage ID. For multiple passages, separate IDs with commas (e.g., `<|1,3|>`).

3. **Avoid speculation**  
   - Do not include information, interpretations, or assumptions beyond what is explicitly stated in the passages.  

4. **Use concise phrasing**  
   - Answer directly, without introductory phrases like "Based on the text" or "According to the passage."

5. **Answer language**
    - Unless the user specifies otherwise, your response should be in the same language as the user's question.

### Citation Rules  
1. **Citation format**  
   - Use `<|N|>` to reference passage IDs for all supporting evidence. Combine multiple passage IDs with commas (e.g., `<|1,2|>`).  

2. **Selective citation**  
   - If the answer can be derived from a single passage, cite only that passage.  
   - If the answer requires combining information from multiple passages, cite all relevant passages.  

3. **When no evidence is available**  
   - If no relevant evidence exists, state:  
     "The text does not provide sufficient evidence to answer this question."

### Output Example  
```
Statement 1<|1|>
Statement 2<|2,3|>
```

</instructions>
"""


reading_user_prompt = """\
Please strictly follow the rules outlined in the instructions. Ensure your answers are accurate, properly cited, and based only on the provided passages. Avoid any extrapolation, assumptions, or guesswork.

### Task  
{text}
"""

reading_sys_prompt = """\
<role>
你是一名阅读理解专家，专长于根据文本片段进行精准解读与分析。
</role>

<instructions>
你的任务是依照提供的文本片段回答后续的问题，必须遵循以下严格规定：

【基本要求】
1. **依据文本片段答题**  
   - 答案只能基于文本片段内的明确信息。  
   - 若文本片段无法充分回答问题，需进行概括说明并补充：“根据已有的信息不足以回答此问题，请补充背景知识。”

2. **准确引用证据**  
   - 回答中的每个关键信息都必须用文本中提供的证据支持。  
   - 使用格式 `<|N|>` 来引用文本片段，其中 N 代表对应的 passage id；如有多个支持的文本，请以逗号分隔（例如 `<|1,3|>`）。

3. **严格避免推测或扩展**  
   - 不允许加入文本片段外的任何信息。  
   - 不得进行主观推断、解释或额外扩展分析。

4. **回答措辞注意**  
   - 直接回答问题，避免使用诸如“根据文本片段”、“根据提供的信息”或“Based on the provided text”等句首提示表述。 

5. **回答语言**
    - 除非用户要求，否则你回答的语言需要和用户提问的语言保持一致。

【引用规则】
1. **引用格式**  
   - 每个支持回答的文本必须以 `<|N|>` 格式标明，其中 N 是对应文本的 passage id。  
   - 同时支持回答的多个文本用逗号分隔，如 `<|1,2|>`。

2. **引用选择性**  
   - 只引用直接与回答相关的文本段落，不可引用与答案不直接相关的内容。

3. **无相关证据时**  
   - 若文本片段中无法找到直接证据，需说明“文本中缺乏足够证据支持该问题的回答。”，不得进行臆测。

【输出示例】
```
陈述1<|1|>
陈述2<|2,3|>
```

</instructions>
"""

reading_user_prompt = """\
请严格遵守以上规则，确保答案准确、引用正确，不作扩展或猜测。下面是具体任务：

{xml_template}"""



import json
import time
import concurrent.futures
from dataclasses import dataclass
from loguru import logger
from typing import Optional, Dict, Any, List, Union
from openai import OpenAI
from tqdm import tqdm

@dataclass
class Config:
    api_key: str = "sk-KGAYIrOrysbLkOzoEPqRD1NwfhBvG2DPtjXTPNEoGeI70S8v"
    base_url: str = "https://api.pandalla.ai/v1"
    gpt_model: str = "o3"
    claude_model: str = "claude-3-7-sonnet-20250219"
    gemini_model: str = "gemini-2.5-pro-exp-03-25"
    grok_model: str = "grok-3-all"
    input_path: str = "/home/tom/fssd/yourbench/chemistry_train/multi_hop_questions.json"
    output_path: str = "/home/tom/fssd/yourbench/chemistry_train/multi_hop_questions_sft.json"
    max_retries: int = 3
    retry_delay: int = 2
    api_call_delay: float = 1
    prefix: str = "根据以下检索得到的文本片段，回答后续问题。回答时引用用于回答的文本片段。\n\n## 文本片段\n"
    suffix: str = "## 后续问题\n"
    nichs: str = "请严格遵守以上规则，确保答案准确、引用正确，不作扩展或猜测。下面是具体任务：\n\n"
    max_workers: int = 4

config = Config()


def clear_document_name(document: str) -> str:
    """清除文档名称中的格式后缀"""
    formats = [".pdf", ".md", ".html", ".txt", ".doc", ".docx"]
    for fmt in formats:
        if document.lower().endswith(fmt):
            return document[:-len(fmt)].rstrip('.')
    return document

def construct_passages_xml(chunks: Union[List[str], str], question: str, document: str) -> str:
    """构建包含上下文的XML格式查询"""
    passages = []
    if isinstance(chunks, str):
        chunks = [chunks]
    
    for i, chunk in enumerate(chunks):
        file_name = f"<file_name>{clear_document_name(document)}</file_name>"
        section_text = chunk[-6:-1] if len(chunk) >= 5 else chunk
        file_section = f"<file_section>{section_text}</file_section>"
        
        passage = f"""<passage id={i}>
{file_name}
{file_section}
<content>{chunk}</content>
</passage>"""
        passages.append(passage)
    
    context = "<passages>\n" + "\n\n".join(passages) + "\n</passages>"
    
    xml_template = f"""根据以下检索得到的文本片段，回答后续问题。回答时引用用于回答的文本片段。

## 文本片段
{context}

## 后续问题
<question>{question}</question>"""
    
    return xml_template

def get_model_response(model: str, system_prompt: str, user_prompt: str, 
                     max_retries: Optional[int] = None, 
                     retry_delay: Optional[int] = None) -> Optional[str]:
    """
    调用 API 获取指定模型的响应并进行重试。
    """
    max_retries = max_retries if max_retries is not None else config.max_retries
    retry_delay = retry_delay if retry_delay is not None else config.retry_delay

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            )
            # 添加API调用间隔
            time.sleep(config.api_call_delay)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"{model} API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避策略
            else:
                return None
    return None

def answer_type_mapping(item:dict):
    if len(item['self_answer']) > 100:
        return "段落"
    else:
        return "句子"

def process_item(item):
    data = {
        "id": item['question'],
        "conversations": [
            {
                "from": "user",
                "value": ""
            },
            {
                "from": "assistant",
                "value": ""
            }
        ],
    "type": "L1： 单篇章事实类",
    "task_parent": "问答",
    "priority": "P0",
    "docs": "单文档",
    "area": "化学",
    "conv_turns": 1,
    "chunk_lang": "中文",
    "chunk_nums": 1,
    "question_lang": "中文",
    "question_type": "文本",
    "output_type": "有答案",
    "answer_lang": "中文",
    "answer_type": answer_type_mapping(item),
    "task_child": "阅读理解"
}
    try:
        chunks = item.get("source_chunk_texts", item.get("chunk_text", []))
        question = item["question"]
        document = item["document"]
        xml_template = construct_passages_xml(chunks, question, document)
        user_prompt = reading_user_prompt.format(xml_template=xml_template)
        data["conversations"][0]["value"] = user_prompt.replace(config.nichs, "")
        # 获取GPT模型回答
        gpt_response = get_model_response(config.gpt_model, reading_sys_prompt, user_prompt)
        if gpt_response:
            data["conversations"][1]["value"] = gpt_response
        else:
            data["conversations"][1]["value"] = "API调用失败，未能获取回答"
            logger.warning(f"GPT处理项目失败: {item.get('question', '未知question')}")
        
        return data
    except Exception as e:
        logger.error(f"处理项目时发生错误: {e}")
        return data

def process_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量处理数据项"""
    results = []
    with tqdm(total=len(items), desc="处理进度") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in items}
            for future in concurrent.futures.as_completed(future_to_item):
                results.append(future.result())
                pbar.update(1)
    return results


def main():   
    # 初始化需要的全局变量
    global client
    
    # 初始化API客户端
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    
    
    # 加载数据
    logger.info(f"正在从 {config.input_path} 加载数据...")
    with open(config.input_path, 'r') as f:
        data = json.load(f)
    
    # data = data[:4] 
    
    logger.info(f"成功加载数据，共 {len(data)} 条")
    
    # 处理数据
    results = process_batch(data)

    # 保存结果
    logger.info(f"正在将结果保存到 {config.output_path}")
    with open(config.output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    logger.success(f"处理完成，共处理 {len(results)} 条数据")

if __name__ == "__main__":
    main()