from typing import List, Optional
import numpy as np
from ..core.types import SensorRecord, ExtractedFeatures, PredictionResult
from ..ai.provider import AIProvider
from .visualizer import Visualizer
from ..core.knowledge_base import KnowledgeBase

class BatchAnalyst:
    def __init__(self, ai_provider: Optional[AIProvider], output_dir: str):
        self.ai = ai_provider
        self.visualizer = Visualizer(output_dir)
        self.kb = KnowledgeBase()

    def generate_full_report(self, 
                             batch_id: str, 
                             records: List[SensorRecord], 
                             final_feats: ExtractedFeatures, 
                             final_pred: PredictionResult, 
                             real_capacity: float):
        
        # 1. Generate Visualizations first
        self.visualizer.generate_report_charts(batch_id, records)

        # 2. Analyze with AI (RAG Enhanced)
        analysis_text = "AI Provider Not Available"
        if self.ai:
            print("[Report] Retrieving Context from Vector DB...")
            # RAG: Get similar cases
            query_context = f"Temp={final_feats.temp_mean:.1f}C, pH_slope={final_feats.ph_slope:.4f}"
            similar_cases = self.kb.find_similar_cases(query_context)
            
            print("[Report] Invoking AI Agent for In-depth Analysis...")
            
            # Simple stats
            ph_start = records[0].ph
            ph_end = records[-1].ph
            temp_max = max(r.temperature for r in records)
            cond_final = records[-1].conductivity
            
            prompt = f"""
            Role: Expert Chemical Engineer.
            Task: Assess batch quality and Recommend parameters for NEXT batch.
            
            Current Batch: {batch_id}
            - Duration: {records[-1].time_min} min
            - pH: {ph_start:.2f} -> {ph_end:.2f}
            - Temp: Mean {final_feats.temp_mean:.1f}°C, Max {temp_max:.1f}°C
            - Pred Capacity: {final_pred.capacity:.2f} mmol/g (True: {real_capacity:.2f})
            
            {similar_cases}
            
            Please provide:
            1. **Diagnosis**: Why did it succeed/fail?
            2. **Optimization**: Specific parameter changes for the next experiment (e.g. "Lower Target Temp by 50C").
            """
            
            try:
                analysis_text = self.ai.generate(prompt)
                
                # Auto-Index this result for future RAG
                quality = "good" if real_capacity > 2.0 else "bad"
                self.kb.add_experiment_insight(batch_id, analysis_text[:500], quality)
                
            except Exception as e:
                analysis_text = f"AI Analysis Failed: {str(e)}"
        
        # 3. Print/Save Report (For now just print to console as requested)
        print("\n" + "#"*60)
        print(f"   ACADEMIC BATCH REPORT: {batch_id}")
        print("#"*60 + "\n")
        print(analysis_text)
        print("\n" + "#"*60)
        
        return analysis_text
