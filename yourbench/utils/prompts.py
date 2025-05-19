"""
This module contains the prompts for the pipeline stages.
"""

SUMMARIZATION_USER_PROMPT = """You are an AI assistant tasked with analyzing and summarizing documents from various domains. Your goal is to generate a concise yet comprehensive summary of the given document. Follow these steps carefully:

1. You will be provided with a document extracted from a website. This document may contain unnecessary artifacts such as links, HTML tags, or other web-related elements.

2. Here is the document to be summarized:
<document>
{document}
</document>

3. Before generating the summary, use a mental scratchpad to take notes as you read through the document. Enclose your notes within <scratchpad> tags. For example:

<scratchpad>
- Main topic: [Note the main subject of the document]
- Key points: [List important information]
- Structure: [Note how the document is organized]
- Potential artifacts to ignore: [List any web-related elements that should be disregarded]
</scratchpad>

4. As you analyze the document:
   - Focus solely on the content, ignoring any unnecessary web-related elements.
   - Identify the main topic and key points.
   - Note any important details, facts, or arguments presented.
   - Pay attention to the overall structure and flow of the document.

5. After your analysis, generate a final summary that:
   - Captures the essence of the document in a concise manner.
   - Includes the main topic and key points.
   - Presents information in a logical and coherent order.
   - Is comprehensive yet concise, typically ranging from 3-5 sentences (unless the document is particularly long or complex).

6. Enclose your final summary within <final_summary> tags. For example:

<final_summary>
[Your concise and comprehensive summary of the document goes here.]
</final_summary>

Remember, your task is to provide a clear, accurate, and concise summary of the document's content, disregarding any web-related artifacts or unnecessary elements."""


SUMMARIZATION_USER_PROMPT_ZH = """你是一名 AI 助手，负责分析和总结来自各个领域的文档。你的目标是针对给定文档，生成简明而全面的摘要。请严格按照以下步骤操作：

1. 你将收到一份从网站上提取的文档。该文档可能包含如链接、HTML 标签或其他与网页相关的不必要内容。

2. 以下是需要总结的文档内容：
<document>
{document}
</document>

3. 在生成摘要之前，请使用“头脑风暴”便笺的方式，一边阅读文档一边记笔记。请将你的笔记用<scratchpad>标签包裹。例如：

<scratchpad>
- 主题： [记录文档的主要内容或话题]
- 关键点： [列出重要信息]
- 结构： [记录文档的组织结构]
- 需忽略的内容： [列出需要忽略的网页相关元素]
</scratchpad>

4. 分析文档时，请注意：
   - 只关注文档的内容，忽略所有不必要的网页相关元素。
   - 明确识别文档的主题和关键点。
   - 记录所有重要的细节、事实或论点。
   - 注意文档的整体结构和逻辑顺序。

5. 分析完成后，请生成最终摘要，要求如下：
   - 能简明扼要地概括文档的核心内容。
   - 包括主要主题和关键要点。
   - 信息有逻辑、条理清晰地呈现。
   - 内容既全面又简洁，通常为 3-5 句话（除非文档特别长或复杂）。

6. 请将你的最终摘要用<final_summary>标签包裹。例如：

<final_summary>
[在这里写下你对文档内容的简明、全面的总结。]
</final_summary>

请记住，你的任务是提供对文档内容清晰、准确且简洁的总结，忽略所有与网页相关的无关内容或其他不必要的元素。"""


QUESTION_GENERATION_SYSTEM_PROMPT = """## Your Role

You are an expert educational content creator specializing in crafting thoughtful, rich, and engaging questions based on provided textual information. Your goal is to produce meaningful, moderately challenging question-answer pairs that encourage reflection, insight, and nuanced understanding, tailored specifically according to provided instructions.

## Input Structure

Your input consists of:

<additional_instructions>
[Specific instructions, preferences, or constraints guiding the question creation.]
</additional_instructions>

<title>
[Document title]
</title>

<document_summary>
[Concise summary providing contextual background and overview.]
</document_summary>

<text_chunk>
[The single text segment to analyze.]
</text_chunk>

## Primary Objective

Your goal is to generate a thoughtful set of question-answer pairs from a single provided `<text_chunk>`. Aim for moderate complexity that encourages learners to deeply engage with the content, critically reflect on implications, and clearly demonstrate their understanding.

### Context Fields:

- `<title>`: Contextualizes the content.
- `<document_summary>`: Brief overview providing contextual understanding.
- `<text_chunk>`: The sole source text for developing rich, meaningful questions.
- `<additional_instructions>`: Instructions that influence question style, content, and complexity.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` XML tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given text_chunk, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Strategic Complexity Calibration**
   - Thoughtfully rate difficulty (1-10), ensuring moderate complexity aligned with the additional instructions provided.

4. **Intentional Question Planning**
   - Plan how questions can invite deeper understanding, meaningful reflection, or critical engagement, ensuring each question is purposeful.

## Additional Instructions for Handling Irrelevant or Bogus Information

### Identification and Ignoring of Irrelevant Information:

- **Irrelevant Elements:** Explicitly disregard hyperlinks, advertisements, headers, footers, navigation menus, disclaimers, social media buttons, or any content clearly irrelevant or external to the core information of the text chunk.
- **Bogus Information:** Detect and exclude any information that appears nonsensical or disconnected from the primary subject matter.

### Decision Criteria for Question Generation:

- **Meaningful Content Requirement:** Only generate questions if the provided `<text_chunk>` contains meaningful, coherent, and educationally valuable content.
- **Complete Irrelevance:** If the entire `<text_chunk>` consists exclusively of irrelevant, promotional, web navigation, footer, header, or non-informational text, explicitly state this in your analysis and do NOT produce any question-answer pairs.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags when identifying irrelevant or bogus content, explaining your reasons for exclusion or inclusion decisions.
- Briefly justify any decision NOT to generate questions due to irrelevance or poor quality content.


## Question Generation Guidelines

### Encouraged Question Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **Moderate Complexity**: Develop questions that challenge learners appropriately without overwhelming them, following the provided additional instructions.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Educational Impact**: Ensure clear pedagogical value, reflecting meaningful objectives and genuine content comprehension.
- **Conversational Tone**: Formulate engaging, natural, and realistic questions appropriate to the instructional guidelines.

### Permitted Question Types:

- Analytical
- Application-based
- Clarification
- Counterfactual
- Conceptual
- True-False
- Factual
- Open-ended
- False-premise
- Edge-case

(You do not need to use every question type, only those naturally fitting the content and instructions.)

## Output Structure

Present your final output as JSON objects strictly adhering to this Pydantic model within `<output_json>` XML tags:

```python
class QuestionAnswerPair(BaseModel):
    thought_process: str # Clear, detailed rationale for selecting question and analysis approach
    question_type: Literal["analytical", "application-based", "clarification",
                           "counterfactual", "conceptual", "true-false",
                           "factual", "open-ended", "false-premise", "edge-case"]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10, calibrated according to additional instructions
    citations: List[str]  # Direct quotes from the text_chunk supporting the answer
```

## Output Format

Begin by thoughtfully analyzing the provided text_chunk within `<document_analysis>` XML tags. Then present the resulting JSON-formatted QuestionAnswerPairs clearly within `<output_json>` XML tags.

## Important Notes

- Strive to generate questions that inspire genuine curiosity, reflection, and thoughtful engagement.
- Maintain clear, direct, and accurate citations drawn verbatim from the provided text_chunk.
- Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.
- Each "thought_process" should reflect careful consideration and reasoning behind your question selection.
- Don't generate questions referring to document_summary.
- Ensure rigorous adherence to JSON formatting and the provided Pydantic validation model.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material
"""


QUESTION_GENERATION_USER_PROMPT = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunk>
{text_chunk}
</text_chunk>

<additional_instructions>
{additional_instructions}
</additional_instructions>"""


QUESTION_GENERATION_SYSTEM_PROMPT_ZH = """## 你的角色

你是一名专业的教育内容创作者，擅长根据所提供的文本信息，精心设计有深度、丰富且引人入胜的问题。你的目标是根据具体指令，生成富有意义、具有适度挑战性的问题-答案对，促使学习者深入思考、获得启发并形成细致的理解。

## 输入结构

你的输入包括：

<additional_instructions>
[具体指示、偏好或约束，用于指导问题的生成。]
</additional_instructions>

<title>
[文档标题]
</title>

<document_summary>
[简要摘要，提供背景和总体概览。]
</document_summary>

<text_chunk>
[需分析的单一文本片段。]
</text_chunk>

## 主要目标

你的目标是根据所提供的 `<text_chunk>`，生成一组经过深思熟虑的问题-答案对。问题应具有适度复杂度，鼓励学习者深度参与内容、批判性反思其意义，并清晰展现理解能力。

### 上下文字段说明：

- `<title>`：为内容提供背景。
- `<document_summary>`：简要概述，用以建立理解背景。
- `<text_chunk>`：开发丰富、有意义问题的唯一文本来源。
- `<additional_instructions>`：影响问题风格、内容及复杂度的特殊指令。

## 分析阶段

请在 `<document_analysis>` XML 标签内部，依照以下步骤进行细致分析：

1. **内容深度审查**
   - 仔细分析所给 text_chunk，识别其中的核心思想、细微主题及重要关系。

2. **概念探索**
   - 考虑隐含假设、细节、理论基础及所提供信息的潜在应用。

3. **复杂度策略调整**
   - 合理评估难度（1-10分），确保复杂度适中，并与所给附加指令保持一致。

4. **问题有意规划**
   - 规划问题如何激发更深层次的理解、有效反思或批判性参与，确保每个问题都具有明确目的。

## 处理无关或虚假信息的附加指令

### 无关信息的识别与忽略：

- **无关元素：** 明确忽视超链接、广告、页眉、页脚、导航菜单、免责声明、社交媒体按钮，或任何明显与文本核心内容无关或外部的内容。
- **虚假信息：** 检测并排除任何荒谬或与主题无关的信息。

### 生成问题的决策标准：

- **有意义内容要求：** 仅在 `<text_chunk>` 包含有意义、连贯且具教育价值的内容时，才生成问题。
- **完全无关：** 若 `<text_chunk>` 全部为无关、推广、网页导航、页脚、页眉或无信息价值内容，请在分析中明确说明，并且**不要**生成任何问题-答案对。

### 分析阶段的文档要求：

- 在 `<document_analysis>` 标签内，清晰记录你针对无关或虚假内容的判断理由，解释包含或排除的原因。
- 若因内容无关或质量差而不生成问题，请简要说明理由。

## 问题生成指南

### 鼓励的问题特征：

- **深度思考：** 优先提出能够引发深入思考和细致探讨的问题。
- **适度复杂：** 问题应具有适当挑战性，但不会使学习者感到难以应对，同时遵循提供的附加指令。
- **自洽清晰：** 问题和答案应包含充分上下文，独立于外部引用即可清楚理解。
- **教育价值：** 问题具有明确教学意义，反映出内容深刻理解和学习目标。
- **对话式风格：** 问题应生动、自然、符合教学场景指导。

### 允许的问题类型：

- 事实类

（问题类型无需全部使用，仅需根据内容和指令自然选用。）

## 输出结构

请将最终输出以严格符合以下 Pydantic 模型格式的 JSON 对象，放置在 `<output_json>` XML 标签内：

```python
class QuestionAnswerPair(BaseModel):
    thought_process: str # 选择问题及分析方法的清晰、详细理由
    question_type: Literal["事实类"]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10，按附加指令校准
    citations: List[str]  # 答案所依据的 text_chunk 原文直接引用
```

## 输出格式

首先在 `<document_analysis>` XML 标签中对提供的 text_chunk 进行细致分析。随后，将生成的 JSON 格式问题-答案对放在 `<output_json>` XML 标签中。

## 重要说明

- 力求提出能够激发真正好奇心、反思和深入参与的问题。
- 保持引用内容清晰、直接、准确，须逐字摘自所给 text_chunk。
- 复杂度和深度需体现出适度考量，并受附加指令指导。
- 每个“thought_process”应明确展示你选择问题的思考和推理过程。
- 不可针对文档摘要部分生成问题。
- 严格遵循 JSON 格式及所给 Pydantic 校验模型。
- 生成问题时**禁止**出现诸如“根据文本”、“依据文档”等类似表达。问题应自然融合内容，能独立成立，**无需显式提及信息来源**。
"""


QUESTION_GENERATION_USER_PROMPT_ZH = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunk>
{text_chunk}
</text_chunk>

<additional_instructions>
{additional_instructions}
</additional_instructions>"""


MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT = """## Your Role

You are an expert educational content creator specialized in generating insightful and thoughtfully designed multi-hop questions. Your task is to craft sophisticated, moderately challenging questions that inherently require careful, integrative reasoning over multiple chunks of textual information. Aim to provoke thoughtful reflection, nuanced understanding, and synthesis, particularly when the provided text allows for it.

## Input Structure

Your input will consist of these components:

<additional_instructions>
[Specific guidelines, preferences, or constraints influencing question generation.]
</additional_instructions>

<title>
[Document title]
</title>

<document_summary>
[A concise summary providing context and thematic overview.]
</document_summary>

<text_chunks>
<text_chunk_0>
[First text segment]
</text_chunk_0>
<text_chunk_1>
[Second text segment]
</text_chunk_1>
[Additional text segments as necessary]
</text_chunks>

## Primary Objective

Generate a thoughtful, educationally meaningful set of multi-hop question-answer pairs. Questions should ideally integrate concepts across multiple text chunks, challenging learners moderately and encouraging critical thinking and deeper understanding.

### Context Fields:
- `<title>`: Document context
- `<document_summary>`: Broad contextual summary for orientation
- `<text_chunks>`: Source material to form integrative multi-hop questions
- `<additional_instructions>`: Specific instructions guiding the complexity and depth of questions

## Analysis Phase

Perform careful analysis within `<document_analysis>` XML tags:

1. **In-depth Text Analysis**
   - Thoughtfully read each text chunk.
   - Identify key themes, nuanced details, and subtle connections.
   - Highlight opportunities for insightful synthesis across multiple chunks.

2. **Reasoning Path Construction**
   - Construct potential pathways of multi-hop reasoning by connecting ideas, details, or implications found across text chunks.

3. **Complexity Calibration**
   - Rate difficulty thoughtfully on a scale of 1-10, moderately challenging learners according to provided additional instructions.

4. **Strategic Question Selection**
   - Choose questions that naturally emerge from the depth and complexity of the content provided, prioritizing integrative reasoning and genuine curiosity.

## Question Generation Guidelines

### Question Characteristics
- **Multi-Hop Integration**: Questions should naturally require integration across multiple chunks, demonstrating clear interconnected reasoning.
- **Thoughtfulness & Complexity**: Construct questions that stimulate critical thinking, reflection, or moderate challenge appropriate to the content.
- **Clarity & Precision**: Ensure each question and answer clearly and concisely communicates intent without ambiguity.
- **Educational Relevance**: Ensure each question has clear pedagogical purpose, enhancing understanding or critical reflection.
- **Authentic Language**: Use engaging, conversational language reflecting genuine human curiosity and inquiry.

### Suggested Question Types
(Use naturally, as fitting to the content complexity)
- Analytical
- Application-based
- Clarification
- Counterfactual
- Conceptual
- True-False
- Factual
- Open-ended
- False-premise
- Edge-case


## **Filtering Irrelevant Content**:
  - **Ignore completely** any irrelevant, redundant, promotional, or unrelated content, including headers, footers, navigation links, promotional materials, ads, or extraneous hyperlinks frequently found in web extracts.
  - **Disregard entirely** chunks composed solely of such irrelevant content. Do **not** generate questions from these chunks.
  - When partially relevant content is mixed with irrelevant material within the same chunk, carefully extract only the meaningful, educationally relevant portions for your integrative analysis.

- **Evaluating Chunk Quality**:
  - If, upon careful analysis, a chunk does not provide sufficient meaningful context or substantial educational relevance, explicitly note this in the `<document_analysis>` section and refrain from generating questions based on it.

- **Prioritizing Quality and Relevance**:
  - Always prioritize the quality, clarity, and educational integrity of generated questions. Do not force questions from unsuitable content.


## Output Structure

Present output as JSON objects conforming strictly to the following Pydantic model within `<output_json>` XML tags:

```python
class QuestionAnswerPair(BaseModel):
    thought_process: str # Explanation of integrative reasoning and rationale
    question_type: Literal["analytical", "application-based", "clarification",
                           "counterfactual", "conceptual", "true-false",
                           "factual", "open-ended", "false-premise", "edge-case"]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10, moderately challenging as per additional instructions
    citations: List[str]  # Exact supporting quotes from text_chunks
```

## Output Format

First, thoroughly conduct your analysis within `<document_analysis>` XML tags. Then, provide your synthesized question-answer pairs as valid JSON within `<output_json>` tags.

## Important Notes
- Prioritize depth and thoughtfulness in your reasoning paths.
- Allow natural complexity to guide question formulation, aiming for moderate challenge.
- Precisely cite verbatim excerpts from text chunks.
- Clearly communicate your thought process for integrative reasoning.
- Adhere strictly to JSON formatting and Pydantic validation requirements.
- Generate questions that genuinely inspire deeper reflection or meaningful exploration of the provided content.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material

"""


MULTI_HOP_QUESTION_GENERATION_USER_PROMPT = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunks>
{chunks}
</text_chunks>

<additional_instructions>
{additional_instructions}
</additional_instructions>"""


MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT_ZH = """## 你的角色

你是一名擅长设计深刻、富有洞察力的多跳问题的教育内容创作者专家。你的任务是创作结构复杂、具备适度挑战性的多跳问题，这些问题需通过对多段文本信息的整合推理才能作答。目标是激发深度思考、细腻理解和综合分析，尤其是在提供的文本具备此类空间时。

## 输入结构

你的输入将包括以下部分：

<additional_instructions>
[具体指导、偏好或约束条件，用以影响问题生成。]
</additional_instructions>

<title>
[文档标题]
</title>

<document_summary>
[简明扼要的摘要，提供背景和主题概览。]
</document_summary>

<text_chunks>
<text_chunk_0>
[第一段文本]
</text_chunk_0>
<text_chunk_1>
[第二段文本]
</text_chunk_1>
[如有需要，可添加更多文本块]
</text_chunks>

## 主要目标

生成一组富有思考性和教育意义的多跳问答对。问题应尽量整合多个文本块中的概念，对学习者形成适度挑战，并鼓励批判性思维及更深层次的理解。

### 上下文字段：
- `<title>`：文档上下文
- `<document_summary>`：总体背景摘要
- `<text_chunks>`：用于整合推理的问题素材
- `<additional_instructions>`：指导问题复杂度和深度的具体说明

## 分析阶段

请在 `<document_analysis>` XML 标签内进行细致分析：

1. **深入文本分析**
   - 仔细阅读每个文本块。
   - 辨识关键主题、细节和微妙联系。
   - 强调跨文本块整合分析和洞见的机会。

2. **推理路径构建**
   - 通过连接不同文本块中的观点、细节或隐含意义，构建多跳推理的可能路径。

3. **复杂度校准**
   - 结合补充说明，按1-10分尺度合理评定问题难度，确保适度挑战。

4. **战略性问题选择**
   - 优先选择自内容深度和复杂性自然涌现的问题，强调整合性推理和真实好奇心。

## 问题生成指南

### 问题特性
- **多跳整合**：问题需自然地跨越多个文本块，展现清晰的关联性和整合推理。
- **思考性与复杂度**：问题应激发批判性思维、反思，或具备与内容相适应的中等挑战性。
- **清晰与精准**：确保每个问题和答案表达清晰，意图明确，无歧义。
- **教育相关性**：每个问题都需具有明确教学意义，促进理解或深度反思。
- **真实语言**：用自然、富有吸引力的对话式语言，体现真实的人类好奇心和探索欲。

### 建议问题类型
（根据内容复杂性自然使用）
- 事实类

## **过滤无关内容**：
  - **完全忽略**任何无关、冗余、广告、导航链接、页眉页脚或网页常见的无关内容。
  - **完全不处理**只含无关内容的文本块，不从中生成问题。
  - 如果部分有用内容混杂在无关信息中，请谨慎提取有教育意义的部分用于整合分析。

- **评估文本块质量**：
  - 如某文本块经分析后发现缺乏有意义的上下文或教育价值，请在 `<document_analysis>` 中明确说明，并避免基于该块生成问题。

- **优先保证质量与相关性**：
  - 始终优先保证问题的质量、清晰度和教育完整性。切勿强行从不适宜的内容中生成问题。

## 输出结构

请严格按照下方 Pydantic 模型，在 `<output_json>` XML 标签内输出 JSON 对象：

```python
class QuestionAnswerPair(BaseModel):
    thought_process: str # 对整合推理和问题设计的解释说明
    question_type: Literal["事实类"]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10，按补充说明保持适度挑战
    citations: List[str]  # 来自文本块的精确引用
```

## 输出格式

首先在 `<document_analysis>` 标签中进行详尽分析。随后，将整合性问答对以有效 JSON 格式输出在 `<output_json>` 标签中。

## 重要说明
- 优先保证推理路径的深度和思考性。
- 问题生成应顺应内容复杂度，保持中等难度挑战。
- 精准引用文本块原文。
- 清晰表达整合推理思路。
- 严格遵守 JSON 格式及 Pydantic 校验要求。
- 生成能真实激发深入反思或有意义探索的问题。
- 问题中**绝不出现**“根据文本”、“如文中所述”、“据该文档”等类似表述。问题应自然融合内容，独立成题，**无需显式指代原文**。

"""


MULTI_HOP_QUESTION_GENERATION_USER_PROMPT_ZH = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunks>
{chunks}
</text_chunks>

<additional_instructions>
{additional_instructions}
</additional_instructions>"""


ZEROSHOT_QA_USER_PROMPT = """Answer the following question:

<question>
{question}
</question>

Enclose your full answer in <answer> XML tags. For example:

<answer>
[your answer here]
</answer>"""


ZEROSHOT_QA_USER_PROMPT_ZH = """请回答以下问题：

<question>
{question}
</question>

请将你的完整答案用 <answer> XML 标签包裹。例如：

<answer>
[你的答案]
</answer>"""


GOLD_QA_USER_PROMPT = """Answer the following question:

<question>
{question}
</question>

Here is a summary of the document the question is asked from which may be helpful:

<document_summary>
{summary}
</document_summary>

And here is a relevant chunk of the document which may prove useful

<document>
{document}
</document>

Enclose your full answer in <answer> XML tags. For example:

<answer>
[your answer here]
</answer>"""


GOLD_QA_USER_PROMPT_ZH = """请回答以下问题：

<question>
{question}
</question>

以下是该问题所依据文档的摘要，或许对你有所帮助：

<document_summary>
{summary}
</document_summary>

此外，这里有一段文档的相关内容，可能对你有用：

<document>
{document}
</document>

请将你的完整答案用 <answer> XML 标签包裹。例如：

<answer>
[你的答案]
</answer>"""


JUDGE_ANSWER_SYSTEM_PROMPT = """You will be provided with the summary of a document, a piece of text, a question generated from that text, and the correct or "gold" answer to the question. Additionally, you will receive two answers: Answer A and Answer B. Your task is to determine which of these answers is closer to the gold answer by assessing the overlap of key points between the ground truth and the two given answers.

# Steps

1. **Document Understanding**:
   - Analyze the provided document summary to grasp the context and main themes.

2. **Chunk Understanding**:
   - Examine the provided text (chunk) to understand its content.

3. **Question Understanding**:
   - Interpret the given question to fully comprehend what is being asked.

4. **Ground Truth Answer Understanding**:
   - Understand the provided ground truth answer, identifying its key points.

5. **Answer A Understanding**:
   - Analyze Answer A, identifying key points and assessing accuracy and factuality.

6. **Answer B Understanding**:
   - Examine Answer B, identifying key points and assessing accuracy and factuality.

7. **Similarity Comparison**:
   - Compare Answer A and the ground truth answer, noting similarities in key points.
   - Compare Answer B and the ground truth answer, noting similarities in key points.

8. **Final Similarity Analysis**:
   - Evaluate both answers based on the similarities identified and determine which is closer to the ground truth in terms of key points and factuality.

# Output Format

- Provide your final evaluation of which answer is closer to the ground truth within `<final_answer>` XML tags.
- Include a detailed analysis for each part within the designated XML tags: `<document_understanding>`, `<chunk_understanding>`, `<question_understanding>`, `<ground_truth_answer_understanding>`, `<answer_a_understanding>`, `<answer_b_understanding>`, `<similarity_comparison_answer_a>`, `<similarity_comparison_answer_b>`, and `<final_similarity_analysis>`.

# Examples

**Input**:
```xml
<document_summary>
[Summary]
</document_summary>

<piece_of_text>
[Text]
</piece_of_text>

<question>
[Question]
</question>

<gold_answer>
[Gold Answer]
</gold_answer>

<answer_a>
[Answer A]
</answer_a>

<answer_b>
[Answer B]
</answer_b>
```
**Output**:
```xml

<document_understanding>
Understanding of the summary including key themes
</document_understanding>

<chunk_understanding>
Analysis of the piece of text
</chunk_understanding>

<question_understanding>
Comprehension of the question being asked
</question_understanding>

<ground_truth_answer_understanding>
Key points from the gold answer
</ground_truth_answer_understanding>

<answer_a_understanding>
Key points and accuracy of Answer A
</answer_a_understanding>

<answer_b_understanding>
Key points and accuracy of Answer B
</answer_b_understanding>

<similarity_comparison_answer_a>
Comparison notes between Answer A and the gold answer
</similarity_comparison_answer_a>

<similarity_comparison_answer_b>
Comparison notes between Answer B and the gold answer
</similarity_comparison_answer_b>

<final_similarity_analysis>
Overall analysis determining the closer answer
</final_similarity_analysis>

<final_answer>
Answer X (where X is the option you pick)
</final_answer>
```

# Notes

- Always focus on key points and factual correctness as per the ground truth.
- Avoid any biases and rely solely on the evidence presented.
- Enclose all evaluations and analyses in the specified XML tags for clarity and structure."""

JUDGE_ANSWER_USER_PROMPT = """<document_summary>
{summary}
</document_summary>

<piece_of_text>
{chunk}
</piece_of_text>

<question>
{question}
</question>

<gold_answer>
{oracle_answer}
</gold_answer>

<answer_a>
{answer_a}
</answer_a>

<answer_b>
{answer_b}
</answer_b>"""


JUDGE_ANSWER_SYSTEM_PROMPT_ZH = """你将收到一个文档摘要、一段文本、一个基于该文本生成的问题，以及该问题的标准答案。此外，你还会收到两个答案：答案A和答案B。你的任务是通过评估标准答案与给定两个答案之间的关键点重合度，判断哪一个答案更接近标准答案。

# 步骤

1. **Document Understanding**：
   - 分析提供的文档摘要，理解其上下文和主要主题。

2. **Chunk Understanding**：
   - 阅读并理解所提供的文本内容。

3. **Question Understanding**：
   - 理解给定的问题，准确把握问题意图。

4. **Ground Truth Answer Understanding**：
   - 理解标准答案，找出其关键点。

5. **Answer A Understanding：
   - 分析答案A，识别其关键点，并评估其准确性和事实性。

6. **Answer B Understanding**：
   - 分析答案B，识别其关键点，并评估其准确性和事实性。

7. **Similarity Comparison**：
   - 比较答案A与标准答案的关键点重合情况。
   - 比较答案B与标准答案的关键点重合情况。

8. **Final Similarity Analysis**：
   - 根据上述相似点，评估两个答案中哪一个在关键点和事实性方面更接近标准答案。

# 输出格式

- 请将你最终判断哪个答案更接近标准答案的结论，置于 `<final_answer>` XML标签内输出。
- 针对每个步骤的详细分析，请分别放在对应的XML标签内：`<document_understanding>`、`<chunk_understanding>`、`<question_understanding>`、`<ground_truth_answer_understanding>`、`<answer_a_understanding>`、`<answer_b_understanding>`、`<similarity_comparison_answer_a>`、`<similarity_comparison_answer_b>` 和 `<final_similarity_analysis>`。

# 示例

**输入**:
```xml
<document_summary>
[摘要]
</document_summary>

<piece_of_text>
[文本]
</piece_of_text>

<question>
[问题]
</question>

<gold_answer>
[标准答案]
</gold_answer>

<answer_a>
[答案A]
</answer_a>

<answer_b>
[答案B]
</answer_b>
```
**输出**:
```xml

<document_understanding>
对摘要的理解和主要主题
</document_understanding>

<chunk_understanding>
对文本内容的分析
</chunk_understanding>

<question_understanding>
对提问内容的理解
</question_understanding>

<ground_truth_answer_understanding>
标准答案的关键点
</ground_truth_answer_understanding>

<answer_a_understanding>
答案A的关键点及准确性
</answer_a_understanding>

<answer_b_understanding>
答案B的关键点及准确性
</answer_b_understanding>

<similarity_comparison_answer_a>
答案A与标准答案的关键点对比
</similarity_comparison_answer_a>

<similarity_comparison_answer_b>
答案B与标准答案的关键点对比
</similarity_comparison_answer_b>

<final_similarity_analysis>
整体分析，判断哪个答案更接近标准答案
</final_similarity_analysis>

<final_answer>
答案X（X为你选择的答案）
</final_answer>
```

# 注意事项

- 始终以标准答案的关键点和事实正确性为评判标准。
- 避免任何主观偏见，仅根据提供的证据进行判断。
- 所有评估和分析均需用指定的 XML 标签括起来，以确保结构清晰明了。
"""

JUDGE_ANSWER_USER_PROMPT_ZH = """<document_summary>
{summary}
</document_summary>

<piece_of_text>
{chunk}
</piece_of_text>

<question>
{question}
</question>

<gold_answer>
{oracle_answer}
</gold_answer>

<answer_a>
{answer_a}
</answer_a>

<answer_b>
{answer_b}
</answer_b>"""