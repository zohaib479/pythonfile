from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import matplotlib.pyplot as plt
import numpy as np
import os
import re

app = FastAPI()

def parse_and_generate_values(complexity_str, n):
    complexity_str = complexity_str.lower().replace(" ", "")
    
    if complexity_str in ["1", "o(1)"]:
        return np.ones_like(n), "O(1)"
    elif complexity_str in ["logn", "o(logn)"]:
        return np.log2(n), "O(log n)"
    elif complexity_str in ["n", "o(n)"]:
        return n, "O(n)"
    elif complexity_str in ["nlogn", "o(nlogn)"]:
        return n * np.log2(n), "O(n log n)"
    elif complexity_str in ["n2", "o(n2)"]:
        return n**2, "O(n²)"
    else:
        raise ValueError(f"Unsupported complexity: {complexity_str}")

@app.get("/status")
def get_status():
    return {"message": "Server is running"}

@app.get("/generate-graph")
def generate_graph(complexity: str, n_max: int = 100):
    try:
        os.makedirs("output", exist_ok=True)
        n = np.linspace(1, n_max, 1000)
        n[0] = 1
        
        values, label = parse_and_generate_values(complexity, n)
        
        plt.figure(figsize=(10, 6))
        plt.plot(n, values, label=label, linewidth=2, color='blue')
        plt.xlabel('Input Size (n)')
        plt.ylabel('Number of Operations')
        plt.title(f'Time Complexity Analysis: {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.fill_between(n, values, alpha=0.1, color='blue')

        output_path = "output/time_complexity_graph.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        return FileResponse(output_path, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
