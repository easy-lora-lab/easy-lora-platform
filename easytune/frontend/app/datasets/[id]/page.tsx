"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { Dataset, apiFetch } from "../../../lib/api";
import styles from "../../shared.module.scss";

type DatasetInfoResponse = {
  dataset_id: number;
  dataset_name: string;
  dataset_info_json: Record<string, unknown> | null;
};

export default function DatasetDetailPage() {
  const params = useParams<{ id: string }>();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfoResponse | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const datasetId = Number(params.id);

  async function load() {
    const [datasetData, infoData] = await Promise.all([
      apiFetch<Dataset>(`/api/datasets/${datasetId}`),
      apiFetch<DatasetInfoResponse>(`/api/datasets/${datasetId}/dataset-info`)
    ]);
    setDataset(datasetData);
    setDatasetInfo(infoData);
  }

  useEffect(() => {
    load().catch((err: Error) => setError(err.message));
  }, [datasetId]);

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
        <h2>质检报告</h2>
        <pre className={styles.code}>{JSON.stringify(dataset.report_json, null, 2)}</pre>
      </section>

      <section className={styles.panel}>
        <h2>dataset_info.json</h2>
        <p className={styles.muted}>数据集名：{datasetInfo?.dataset_name || "-"}</p>
        <pre className={styles.code}>{JSON.stringify(datasetInfo?.dataset_info_json, null, 2)}</pre>
      </section>
    </>
  );
}
