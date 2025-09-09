from dataclasses import Field
from typing import Optional, Type
from urllib.parse import urlparse
import os
import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, Extra
import pandas as pd
import chardet
from framehub import FrameHub


class DownloadToolInput(BaseModel):
    download_url: Optional[str] = Field(
    #download_url: str = Field(
        None,
        #default="",
        description="The download URL for file download"
    ),
    dataset_uri: Optional[str] = Field(
    #download_uri: str = Field(
        #default="",
        None,
        description="The uri of the dataset to be retrieved"
    )


class DownloadTool(BaseTool):
    name: str = "file_download"
    description: str = "Download file for a given URL and dataset uri"
    args_schema: Type[BaseModel] = DownloadToolInput

    class Config:
        extra = Extra.allow

    def __init__(self, frame_hub: FrameHub, **data):
        super().__init__(**data)
        self._frame_hub = frame_hub

    def _parse_file(self, ext: str, content: str, dateiname_value: str) -> dict:
        """Internal: determine format and payload from extension."""
        ext = ext.lower()
        if ext == ".csv":
            frame_id = self._frame_hub.load_csv_string(content, "clean_download", True)
            df = self._frame_hub.frames[frame_id]  #

            return {
                "format": "csv",
                "frame_id": frame_id,
                "file name": dateiname_value or "",
                "row_count": df.height,  # data rows only (header NOT counted)
                "header": list(df.columns),
                "preview": self._frame_hub.preview(frame_id, 10),
                # already list[dict]
            }
        return {
            "format": "xml" if ext == ".xml" else "unknown",
            "content": content,
            "file name": dateiname_value or ""
        }

    def _run(self, download_url, dataset_uri):
        # Make this more efficient, load it once!
        print(download_url)
        print(dataset_uri)
        frame_id = ""
        benchmark_dir = "open-data-benchmark"
        data_dir = os.path.join(benchmark_dir, "daten")
        sources_path = os.path.join(benchmark_dir, "sources.csv")
        df = pd.read_csv(sources_path)
        # Find the row(s) where dataset matches
        match = df[df["dataset"] == dataset_uri]
        if not match.empty:
            try:
                # Get the value from column 'dateiname'
                dateiname_value = match["dateiname"].iloc[0]  # first match
                data_path = os.path.join(data_dir, dateiname_value)
                print("Found:", dateiname_value)
                with open(data_path, "rb") as f:
                    raw = f.read()

                detected = chardet.detect(raw)
                content = raw.decode(detected["encoding"])
                # Detect file format from extension
                ext = os.path.splitext(dateiname_value)[-1].lower()
                return self._parse_file(ext, content, None)

            except Exception as err:
                print("Other error occurred:", err)
        else:
            print("Dataset not found. Downloading it!")
            try:
                response = requests.get(download_url)
                response.raise_for_status()
                content = response.content.decode("utf-8", errors="replace")
                # Detect format from URL extension first
                path = urlparse(download_url).path
                ext = os.path.splitext(path)[-1].lower()
                return self._parse_file(ext, content, None)
            except requests.exceptions.HTTPError as err:
                print("HTTP error occurred:", err)
            except requests.exceptions.RequestException as err:
                # Catches other issues like connection errors, timeouts, etc.
                print("Other error occurred:", err)

