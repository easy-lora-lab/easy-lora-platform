"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { Dataset, apiFetch } from "../../../lib/api";
import styles from "../../shared.module.scss";

type DatasetInfoResponse = {
  dataset_id: number;
  dataset_name: string;
  dataset_info_json: Record<string, unknown> | null;
};

type DatasetPreviewResponse = {
  dataset_id: number;
  records: Record<string, unknown>[];
  total_records: number;
};

type DatasetSplitResponse = {
  dataset_id: number;
  train_path: string;
  valid_path: string;
  train_count: number;
  valid_count: number;
};

export default function DatasetDetailPage() {
  const params = useParams<{ id: string }>();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfoResponse | null>(null);
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [validRatio, setValidRatio] = useState("0.1");
  const [seed, setSeed] = useState("42");
  const [splitResult, setSplitResult] = useState<DatasetSplitResponse | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const datasetId = Number(params.id);

  const load = useCallback(async () => {
    const [datasetData, infoData, previewData] = await Promise.all([
      apiFetch<Dataset>(`/api/datasets/${datasetId}`),
      apiFetch<DatasetInfoResponse>(`/api/datasets/${datasetId}/dataset-info`),
      apiFetch<DatasetPreviewResponse>(`/api/datasets/${datasetId}/preview?limit=6`)
    ]);
    setDataset(datasetData);
    setDatasetInfo(infoData);
    setPreview(previewData);
  }, [datasetId]);

  useEffect(() => {
    load().catch((err: Error) => setError(err.message));
  }, [load]);

  async function convert() {
    setError("");
    setNotice("");
    try {
      await apiFetch<Dataset>(`/api/datasets/${datasetId}/convert`, { method: "POST" });
      setNotice("转换完成。");
      await load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function splitDataset() {
    setError("");
    setNotice("");
    setSplitResult(null);
    try {
      const result = await apiFetch<DatasetSplitResponse>(`/api/datasets/${datasetId}/split`, {
        method: "POST",
        body: JSON.stringify({
          valid_ratio: Number(validRatio),
          seed: Number(seed)
        })
      });
      setSplitResult(result);
      setNotice("train/valid 切分完成。");
      await load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  if (!dataset) {
    return <div className={styles.panel}>加载中...</div>;
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>{dataset.name}</h1>
          <p>数据集详情、质检报告和 dataset_info.json。</p>
        </div>
        <div className={styles.actions}>
          <Link className={styles.secondaryButton} href="/datasets">
            返回列表
          </Link>
          <button className={styles.button} disabled={dataset.status === "converted"} onClick={convert} type="button">
            转换
          </button>
        </div>
      </div>

      {notice && <div className={styles.notice}>{notice}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.panel}>
        <h2>基础信息</h2>
        <dl className={styles.keyValue}>
          <dt>ID</dt>
          <dd>{dataset.id}</dd>
          <dt>原始文件</dt>
          <dd>{dataset.original_file_path}</dd>
          <dt>转换文件</dt>
          <dd>{dataset.converted_file_path || "-"}</dd>
          <dt>文件类型</dt>
          <dd>{dataset.file_type}</dd>
          <dt>文件大小</dt>
          <dd>{dataset.file_size} bytes</dd>
          <dt>样本数</dt>
          <dd>{dataset.sample_count}</dd>
          <dt>格式</dt>
          <dd>{dataset.formatting}</dd>
          <dt>质量分</dt>
          <dd>{dataset.quality_score}</dd>
          <dt>状态</dt>
          <dd>{dataset.status}</dd>
        </dl>
      </section>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>格式检查与错误样本提示</h2>
            <p className={styles.muted}>包含行数、空行、格式识别、质量分、错误和警告。</p>
          </div>
        </div>
        <pre className={styles.code}>{JSON.stringify(dataset.report_json, null, 2)}</pre>
      </section>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>样本预览</h2>
            <p className={styles.muted}>展示前 6 条样本，总样本数 {preview?.total_records || 0}。</p>
          </div>
        </div>
        <pre className={styles.code}>{JSON.stringify(preview?.records || [], null, 2)}</pre>
      </section>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>train/valid 切分</h2>
            <p className={styles.muted}>对已转换数据集生成 LLaMA-Factory 可用的切分文件。</p>
          </div>
        </div>
        <div className={styles.splitControls}>
          <div className={styles.field}>
            <label htmlFor="validRatio">valid ratio</label>
            <input id="validRatio" value={validRatio} onChange={(event) => setValidRatio(event.target.value)} />
          </div>
          <div className={styles.field}>
            <label htmlFor="seed">seed</label>
            <input id="seed" value={seed} onChange={(event) => setSeed(event.target.value)} />
          </div>
          <button className={styles.button} disabled={dataset.status !== "converted"} onClick={splitDataset} type="button">
            执行切分
          </button>
        </div>
        {splitResult && (
          <div className={styles.configList}>
            <div>
              <span>train</span>
              <strong>{splitResult.train_count} samples / {splitResult.train_path}</strong>
            </div>
            <div>
              <span>valid</span>
              <strong>{splitResult.valid_count} samples / {splitResult.valid_path}</strong>
            </div>
          </div>
        )}
      </section>

      <section className={styles.panel}>
        <h2>LLaMA-Factory dataset_info.json</h2>
        <p className={styles.muted}>数据集名：{datasetInfo?.dataset_name || "-"}</p>
        <pre className={styles.code}>{JSON.stringify(datasetInfo?.dataset_info_json, null, 2)}</pre>
      </section>
    </>
  );
}
