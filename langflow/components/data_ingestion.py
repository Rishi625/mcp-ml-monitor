"""
Data Ingestion Components for LangFlow ML Monitoring Pipeline

Custom components for ingesting data from various sources:
- CSV/JSON files
- REST APIs
- Streaming sources
- Data validation
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class DataValidationResult(BaseModel):
    """Result of data validation"""
    is_valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    missing_values: Dict[str, int]
    schema_errors: List[str]
    warnings: List[str]


class CSVIngestionComponent:
    """
    LangFlow Component: CSV Data Ingestion
    
    Reads CSV files and converts to DataFrame for monitoring pipeline.
    """
    
    display_name = "CSV Ingestion"
    description = "Load CSV data for ML monitoring"
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
    
    def load(
        self,
        file_path: str,
        delimiter: str = ",",
        date_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Load CSV file and parse columns"""
        
        # Read CSV
        self.data = pd.read_csv(file_path, delimiter=delimiter)
        
        # Parse date columns
        if date_columns:
            for col in date_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_datetime(self.data[col])
        
        # Ensure numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.metadata = {
            "source": file_path,
            "records": len(self.data),
            "columns": list(self.data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "loaded_at": datetime.now().isoformat()
        }
        
        return {
            "data": self.data.to_dict(orient='records'),
            "metadata": self.metadata
        }


class APIIngestionComponent:
    """
    LangFlow Component: API Data Ingestion
    
    Fetches data from REST APIs for real-time monitoring.
    """
    
    display_name = "API Ingestion"
    description = "Fetch data from REST API endpoints"
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
    
    async def fetch(
        self,
        endpoint: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data_path: str = "data"  # JSON path to data array
    ) -> Dict[str, Any]:
        """Fetch data from API endpoint"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            if method.upper() == "GET":
                response = await client.get(endpoint, headers=headers, params=params)
            else:
                response = await client.post(endpoint, headers=headers, json=params)
        
        response.raise_for_status()
        json_data = response.json()
        
        # Extract data from path
        data = json_data
        for key in data_path.split('.'):
            if key:
                data = data.get(key, data)
        
        if isinstance(data, list):
            self.data = pd.DataFrame(data)
        else:
            self.data = pd.DataFrame([data])
        
        return {
            "data": self.data.to_dict(orient='records'),
            "metadata": {
                "source": endpoint,
                "records": len(self.data),
                "fetched_at": datetime.now().isoformat()
            }
        }


class StreamIngestionComponent:
    """
    LangFlow Component: Stream Data Ingestion
    
    Handles streaming data with windowing and batching.
    """
    
    display_name = "Stream Ingestion"
    description = "Process streaming data with windowing"
    
    def __init__(self, window_size: int = 1000, slide_interval: int = 100):
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.buffer: List[Dict[str, Any]] = []
        self.windows_processed = 0
    
    def add_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add record to buffer, return window if complete"""
        self.buffer.append({
            **record,
            "_ingested_at": datetime.now().isoformat()
        })
        
        if len(self.buffer) >= self.window_size:
            window_data = self.buffer[:self.window_size]
            self.buffer = self.buffer[self.slide_interval:]
            self.windows_processed += 1
            
            return {
                "data": window_data,
                "window_id": self.windows_processed,
                "window_size": len(window_data),
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def flush(self) -> Dict[str, Any]:
        """Flush remaining buffer"""
        if self.buffer:
            data = self.buffer.copy()
            self.buffer = []
            return {
                "data": data,
                "window_id": self.windows_processed + 1,
                "window_size": len(data),
                "partial": True
            }
        return {"data": [], "window_size": 0}


class DataValidatorComponent:
    """
    LangFlow Component: Data Validation
    
    Validates incoming data against schema and quality rules.
    """
    
    display_name = "Data Validator"
    description = "Validate data schema and quality"
    
    def __init__(self):
        self.schema: Dict[str, str] = {}
        self.required_columns: List[str] = []
        self.value_ranges: Dict[str, tuple] = {}
    
    def set_schema(
        self,
        schema: Dict[str, str],
        required: Optional[List[str]] = None,
        ranges: Optional[Dict[str, tuple]] = None
    ):
        """Define validation schema"""
        self.schema = schema
        self.required_columns = required or []
        self.value_ranges = ranges or {}
    
    def validate(self, data: List[Dict[str, Any]]) -> DataValidationResult:
        """Validate data against schema"""
        df = pd.DataFrame(data)
        errors = []
        warnings = []
        invalid_count = 0
        
        # Check required columns
        for col in self.required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check data types
        for col, expected_type in self.schema.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column {col} expected numeric, got {actual_type}")
                elif expected_type == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    warnings.append(f"Column {col} may not be datetime format")
        
        # Check value ranges
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    warnings.append(f"{out_of_range} values in {col} outside range [{min_val}, {max_val}]")
                    invalid_count += out_of_range
        
        # Count missing values
        missing = df.isnull().sum().to_dict()
        missing = {k: v for k, v in missing.items() if v > 0}
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            total_records=len(df),
            valid_records=len(df) - invalid_count,
            invalid_records=invalid_count,
            missing_values=missing,
            schema_errors=errors,
            warnings=warnings
        )


# LangFlow component registration
LANGFLOW_COMPONENTS = {
    "csv_ingestion": CSVIngestionComponent,
    "api_ingestion": APIIngestionComponent,
    "stream_ingestion": StreamIngestionComponent,
    "data_validator": DataValidatorComponent
}

