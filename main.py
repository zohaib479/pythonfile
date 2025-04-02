from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
import nest_asyncio
import uvicorn
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Apply nest_asyncio to allow running Uvicorn in Jupyter/Colab
nest_asyncio.apply()

app = FastAPI()

def parse_and_generate_values(complexity_str, n):
    # Clean up the input string
    complexity_str = complexity_str.lower().replace(" ", "")
    
    # Handle special cases and complexity patterns
    if complexity_str in ["1", "c", "constant", "o(1)", "O(1)"]:
        return np.ones_like(n), "O(1)"
        
    elif complexity_str in ["log*n","logn", "log(n)", "o(logn)", "o(log(n))"]:
        return np.log2(n), "O(log n)"
        
    elif complexity_str in ["n*m","m*n","n", "linear", "o(n)"]:
        return n, "O(n)"
        
    elif complexity_str in ["nlogn", "nlog(n)", "n*logn", "n*log(n)", "o(nlogn)"]:
        return n * np.log2(n), "O(n log n)"
        
    elif complexity_str in ["n2", "n^2", "nsquared", "quadratic", "o(n2)", "o(n^2)", "n**2x","n**2"]:
        return n**2, "O(n²)"
        
    elif complexity_str in ["n3", "n^3", "ncubed", "cubic", "o(n3)", "o(n^3)", "n**3"]:
        return n**3, "O(n³)"
        
    # Handle 2^h, 2^n, etc.
    elif re.match(r"2\^[a-zA-Z]", complexity_str) or complexity_str in ["exponential", "o(2^n)", "2**n", "2**h"]:
        label = f"O({complexity_str})" if not complexity_str.startswith("o(") else complexity_str.upper()
        # Limit the values to avoid overflow
        return np.where(n < 30, 2**n, np.inf), label
        
    elif complexity_str in ["n!", "factorial", "o(n!)"]:
        # Use Stirling's approximation for large n to avoid overflow
        return np.where(n < 20, np.array([np.math.factorial(int(i)) for i in n]), np.inf), "O(n!)"
        
    else:
        raise ValueError(f"Unsupported complexity: {complexity_str}")

@app.get("/status")
def get_status():
    return {"message": "Server is running"}

@app.get("/generate-graph")
def generate_graph(
    complexity: str = Query(..., description="Time complexity (e.g., 'nlogn', 'n2', '2^h')"),
    n_max: int = Query(100, description="Maximum input size to plot")
):
    try:
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Generate input sizes (more points for better curve)
        n = np.linspace(1, n_max, 1000)
        n[0] = 1  # Ensure we start at 1 to avoid log(0)
        
        # Generate complexity values and get the formatted label
        values, label = parse_and_generate_values(complexity, n)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(n, values, label=label, linewidth=2, color='blue')
        
        # Find appropriate scale based on data
        if np.any(values > 1000):
            plt.yscale('log')
        
        # Add labels and title
        plt.xlabel('Input Size (n)')
        plt.ylabel('Number of Operations')
        plt.title(f'Time Complexity Analysis: {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add algorithm growth region
        plt.fill_between(n, values, alpha=0.1, color='blue')
        
        # Save the plot
        output_path = "output/time_complexity_graph.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # Return the image file
        return FileResponse(output_path, media_type="image/png")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
