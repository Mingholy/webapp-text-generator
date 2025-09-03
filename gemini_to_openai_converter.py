import json
from typing import Dict, Any, List, Union


def convert_gemini_to_openai_message(gemini_message: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个 Gemini 格式的 message 转换为 OpenAI 格式。

    支持文本和多模态 (图片/文件) 输入。

    Args:
        gemini_message: 来自 Gemini API 的单个 message 字典。
                        通常包含 'role' 和 'parts' 键。
                        'role' 映射: 'user' -> 'user', 'model' -> 'assistant'。
                        'parts' 是一个包含文本或多媒体内容的列表。

    Returns:
        符合 OpenAI API 规范的单个 message 字典。
        包含 'role' 和 'content' 键。'content' 可以是字符串或包含文本/图像 URL 的字典列表。
    """
    openai_message = {}
    
    # 1. 转换角色 (role)
    gemini_role = gemini_message.get('role')
    if gemini_role == 'user':
        openai_message['role'] = 'user'
    elif gemini_role == 'model':
        openai_message['role'] = 'assistant'
    else:
        # Gemini 可能有 'function' 等角色，OpenAI 对应 'tool'，但此处简化处理
        openai_message['role'] = gemini_role 

    # 2. 转换内容 (content)
    gemini_parts = gemini_message.get('parts', [])
    openai_content: List[Dict[str, Any]] = []

    for part in gemini_parts:
        if isinstance(part, str):
            # 如果 part 直接是字符串
            openai_content.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            mime_type = part.get('mimeType', '')
            if mime_type.startswith('text/'):
                # 文本类型数据
                openai_content.append({"type": "text", "text": part.get('text', '')})
            elif mime_type.startswith('image/') or mime_type.startswith('application/'):
                # 图片或文件类型数据 (假设以 base64 编码或提供 URL)
                # Gemini 的 'inlineData' 对应 base64, 'fileData' 对应 Google AI File API 的 URI
                inline_data = part.get('inlineData', {})
                file_data = part.get('fileData', {})
                
                if inline_data:
                    # 处理 base64 编码的内联数据
                    data = inline_data.get('data', '')
                    openai_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{inline_data.get('mimeType', 'image/jpeg')};base64,{data}"
                        }
                    })
                elif file_data:
                    # 处理文件数据 (这里假设 file_uri 可以被直接访问，实际情况可能需要转换)
                    # 注意：OpenAI 通常需要一个可公开访问的 URL 或 base64 数据
                    # 此处为示意，实际应用中可能需要额外处理
                    file_uri = file_data.get('file_uri', '')
                    openai_content.append({
                        "type": "image_url", # 或根据文件类型调整
                        "image_url": {
                            "url": file_uri # 直接使用，可能不适用于所有情况
                        }
                    })
                # 可以根据需要添加对 'function_call' 或 'tool_calls' 的处理
            else:
                # 无法识别的 MIME 类型，尝试获取文本
                text = part.get('text', str(part))
                openai_content.append({"type": "text", "text": text})
        else:
            # 兜底处理，将 part 转为字符串
            openai_content.append({"type": "text", "text": str(part)})

    # 3. 简化 content: 如果只有一个文本部分，则直接使用字符串
    if len(openai_content) == 1 and openai_content[0]['type'] == 'text':
        openai_message['content'] = openai_content[0]['text']
    else:
        openai_message['content'] = openai_content
        
    return openai_message


# --- 示例用法 ---
if __name__ == '__main__':
    # 示例 1: 纯文本消息
    gemini_text_msg = {
        "role": "user",
        "parts": ["Hello, how are you?"]
    }
    
    # 示例 2: 多模态消息 (文本 + 图片 base64)
    gemini_multimodal_msg = {
        "role": "user",
        "parts": [
            "What is this picture of?",
            {
                "mimeType": "image/jpeg",
                "inlineData": {
                    "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/...", # 示例 base64 数据
                    "mimeType": "image/jpeg"
                }
            }
        ]
    }

    # 示例 3: 模型回复
    gemini_model_msg = {
        "role": "model",
        "parts": ["I'm doing well, thank you! The picture shows a cat."]
    }

    print("--- Gemini to OpenAI 转换示例 ---")
    print("输入 (Gemini 文本):", json.dumps(gemini_text_msg, indent=2, ensure_ascii=False))
    print("输出 (OpenAI):", json.dumps(convert_gemini_to_openai_message(gemini_text_msg), indent=2, ensure_ascii=False))
    print("-" * 20)
    print("输入 (Gemini 多模态):", json.dumps(gemini_multimodal_msg, indent=2, ensure_ascii=False))
    print("输出 (OpenAI):", json.dumps(convert_gemini_to_openai_message(gemini_multimodal_msg), indent=2, ensure_ascii=False))
    print("-" * 20)
    print("输入 (Gemini 模型):", json.dumps(gemini_model_msg, indent=2, ensure_ascii=False))
    print("输出 (OpenAI):", json.dumps(convert_gemini_to_openai_message(gemini_model_msg), indent=2, ensure_ascii=False))