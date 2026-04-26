"use client";

import { useEffect, useState } from "react";

import { ValidationRecord, apiFetch } from "../../lib/api";
import styles from "../shared.module.scss";

export default function ValidationRecordsPage() {
  const [records, setRecords] = useState<ValidationRecord[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch<ValidationRecord[]>("/api/validation-records")
      .then((items) => setRecords(items))
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>验收记录</h1>
          <p>人工验收历史。</p>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>ID</th>
              <th>模型版本</th>
              <th>prompt</th>
              <th>expected_answer</th>
              <th>actual_answer</th>
              <th>评分</th>
              <th>备注</th>
              <th>创建时间</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              <tr key={record.id}>
                <td>{record.id}</td>
                <td>{record.model_version_id}</td>
                <td>{record.prompt}</td>
                <td>{record.expected_answer || "-"}</td>
                <td>{record.actual_answer}</td>
                <td>{record.human_score}</td>
                <td>{record.human_note || "-"}</td>
                <td>{new Date(record.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
