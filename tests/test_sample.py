#!/usr/bin/env python3
"""Sampling regressions; set SEQTK to test a different executable."""

import gzip
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SEQTK = Path(os.environ.get("SEQTK", Path(__file__).resolve().parents[1] / "seqtk")).resolve()


def fastq(name, sequence="", quality=None):
    if quality is None:
        quality = "I" * len(sequence)
    return "@{}\n{}\n+\n{}\n".format(name, sequence, quality)


class SampleTest(unittest.TestCase):
    def run_seqtk(self, args, data):
        result = subprocess.run([str(SEQTK)] + args, input=data, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def sample(self, data, amount, seed=11, two_pass=False, compressed=False):
        args = ["sample", "-s" + str(seed)]
        if two_pass:
            args.append("-2")
        if two_pass or compressed:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ("input.fq.gz" if compressed else "input.fq")
                if compressed:
                    with gzip.open(path, "wt") as stream:
                        stream.write(data)
                else:
                    path.write_text(data)
                return self.run_seqtk(args + [str(path), str(amount)], "")
        return self.run_seqtk(args + ["-", str(amount)], data)

    def records(self, output):
        lines = output.splitlines()
        self.assertEqual(len(lines) % 4, 0, output)
        records = []
        for i in range(0, len(lines), 4):
            header, sequence, plus, quality = lines[i:i + 4]
            self.assertTrue(header.startswith("@"), output)
            self.assertEqual(plus, "+", output)
            self.assertEqual(len(sequence), len(quality), output)
            records.append((header[1:], sequence, quality))
        return records

    def test_fractional_sampling_preserves_empty_fastq(self):
        data = fastq("empty") + fastq("other", "AAA", "999")
        self.assertEqual(self.sample(data, 0.9), data)

    def test_fixed_count_preserves_empty_records_in_every_position(self):
        data = fastq("first") + fastq("middle", "ACG") + fastq("last")
        for count in (3, 10):
            for two_pass in (False, True):
                with self.subTest(count=count, two_pass=two_pass):
                    self.assertEqual(self.sample(data, count, two_pass=two_pass), data)

    def test_reservoir_replacement_clears_sequence_and_quality(self):
        data = fastq("first", "AAA", "999") + fastq("empty")
        self.assertEqual(self.sample(data, 1, seed=1), fastq("empty"))

    def test_reservoir_replacement_clears_comment(self):
        data = fastq("first old-comment", "AAA") + fastq("second", "CCC")
        self.assertEqual(self.sample(data, 1, seed=1), fastq("second", "CCC"))

    def test_empty_fasta_records_are_preserved(self):
        data = ">first\n\n>middle\nACG\n>last\n\n"
        for two_pass in (False, True):
            with self.subTest(two_pass=two_pass):
                self.assertEqual(self.sample(data, 10, two_pass=two_pass), data)
        self.assertEqual(self.sample(">first\nAAA\n>empty\n\n", 1, seed=1), ">empty\n\n")

    def test_empty_input_emits_no_unfilled_slots(self):
        for amount, two_pass in ((10, False), (10, True), (0.5, False)):
            with self.subTest(amount=amount, two_pass=two_pass):
                self.assertEqual(self.sample("", amount, two_pass=two_pass), "")

    def test_normal_sampling_keeps_seeded_selection_and_order(self):
        for is_fastq in (False, True):
            records = [fastq("r%d comment%d" % (i, i), "ACGT") if is_fastq
                       else ">r%d comment%d\nACGT\n" % (i, i) for i in range(6)]
            for amount, two_pass, selected in ((3, False, [4, 5, 3]),
                                                (3, True, [3, 4, 5]),
                                                (0.5, False, [0, 2, 4, 5])):
                with self.subTest(is_fastq=is_fastq, amount=amount, two_pass=two_pass):
                    self.assertEqual(self.sample("".join(records), amount, two_pass=two_pass),
                                     "".join(records[i] for i in selected))

    def test_paired_sampling_keeps_ids_and_original_records(self):
        mates = [[fastq("r%d/%d" % (i, mate), "" if (i + mate) % 3 == 0 else "ACGT")
                  for i in range(30)] for mate in (1, 2)]
        for seed in (1, 7, 100):
            for amount, two_pass in ((8, False), (8, True), (0.35, False)):
                with self.subTest(seed=seed, amount=amount, two_pass=two_pass):
                    outputs = [self.sample("".join(records), amount, seed, two_pass)
                               for records in mates]
                    parsed = [self.records(output) for output in outputs]
                    self.assertEqual([r[0].split("/")[0] for r in parsed[0]],
                                     [r[0].split("/")[0] for r in parsed[1]])
                    if amount == 8:
                        self.assertEqual(len(parsed[0]), 8)
                    for originals, output, records in zip(mates, outputs, parsed):
                        for record in records:
                            self.assertIn(fastq(*record), originals)
                        self.assertEqual(output, self.sample("".join(originals), amount, seed, two_pass))
                    if two_pass:
                        ids = [int(r[0].split("/")[0][1:]) for r in parsed[0]]
                        self.assertEqual(ids, sorted(ids))

    def test_gzip_empty_records(self):
        data = fastq("first") + fastq("other", "ACG") + fastq("last")
        for two_pass in (False, True):
            with self.subTest(two_pass=two_pass):
                self.assertEqual(self.sample(data, 3, two_pass=two_pass, compressed=True), data)

    def test_shared_printer_preserves_formats_and_conversions(self):
        fq = fastq("empty") + fastq("other", "ACG")
        fa = ">empty\n\n>other\nACG\n"
        for width in (0, 60):
            for args, data, expected in (([], fq, fq), (["-a"], fq, fa),
                                         (["-F", "I"], fa, fq)):
                with self.subTest(width=width, args=args):
                    self.assertEqual(self.run_seqtk(["seq", "-l", str(width)] + args + ["-"], data),
                                     expected)

    def test_reverse_complement_suffix_printer_preserves_empty_fastq(self):
        self.assertEqual(self.run_seqtk(["seq", "-R", "-"], fastq("empty")),
                         fastq("empty+") + fastq("empty-"))


if __name__ == "__main__":
    unittest.main()
