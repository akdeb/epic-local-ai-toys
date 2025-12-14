from openai import SentenceBuffer

def test_buffer():
    print("Testing SentenceBuffer...")
    buf = SentenceBuffer()
    
    tokens = ["Hello", " world", "!", " How", " are", " you", "?"]
    
    print(f"Feeding tokens: {tokens}")
    
    for token in tokens:
        sentences = buf.add(token)
        if sentences:
            print(f"Token '{token}' triggered sentences: {sentences}")
        else:
            print(f"Token '{token}' -> buffered")
            
    remaining = buf.flush()
    if remaining:
        print(f"Flushed: '{remaining}'")

if __name__ == "__main__":
    test_buffer()
