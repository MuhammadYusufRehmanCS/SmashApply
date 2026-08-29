import { Document, Page, StyleSheet, Text } from "@react-pdf/renderer";

const styles = StyleSheet.create({
  page: { padding: 40, fontSize: 11, lineHeight: 1.5, fontFamily: "Helvetica" },
  line: { marginBottom: 2 },
});

export function CvPdfDocument({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <Document>
      <Page size="A4" style={styles.page}>
        {lines.map((line, i) => (
          <Text key={i} style={styles.line}>
            {line || " "}
          </Text>
        ))}
      </Page>
    </Document>
  );
}
