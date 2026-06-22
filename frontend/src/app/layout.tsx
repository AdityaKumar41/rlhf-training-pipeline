import "./styles.css";

export const metadata = {
  title: "RLHF Annotation",
  description: "Preference labeling workspace"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

