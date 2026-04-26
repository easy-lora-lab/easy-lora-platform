import Link from "next/link";
import type { Metadata } from "next";

import "./globals.scss";
import styles from "./shared.module.scss";

export const metadata: Metadata = {
  title: "EasyTune",
  description: "EasyTune 微调傻瓜包"
};

const navItems = [
  { href: "/datasets", label: "数据集管理" },
  { href: "/training-jobs/new", label: "创建训练任务" },
  { href: "/training-jobs", label: "训练任务" },
  { href: "/model-versions", label: "模型版本" },
  { href: "/validation", label: "人工验收" },
  { href: "/validation-records", label: "验收记录" }
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <div className={styles.shell}>
          <header className={styles.header}>
            <Link className={styles.brand} href="/">
              <strong>EasyTune</strong>
              <span>围绕 LLaMA-Factory / RWKV 的私有化微调外壳</span>
            </Link>
            <nav className={styles.nav}>
              {navItems.map((item) => (
                <Link key={item.href} href={item.href}>
                  {item.label}
                </Link>
              ))}
            </nav>
          </header>
          <main className={styles.main}>{children}</main>
        </div>
      </body>
    </html>
  );
}
