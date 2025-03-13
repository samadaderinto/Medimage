"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface WordByWordTextProps {
  text: string;
  delay?: number; // Delay between words (in ms)
}

export default function WordByWordText({
  text,
  delay = 200,
}: WordByWordTextProps) {
  const words = text.split(" ");
  const [visibleCount, setVisibleCount] = useState(0);

  useEffect(() => {
    let i = 0;
    setVisibleCount(0); // Reset visible words on text change

    const interval = setInterval(() => {
      if (i < words.length) {
        setVisibleCount((prev) => prev + 1);
        i++;
      } else {
        clearInterval(interval);
      }
    }, delay);

    return () => clearInterval(interval);
  }, [text, delay]);

  return (
    <div className="font-medium text-gray-800 pt-3">
      {words.slice(0, visibleCount).map((word, index) => (
        <motion.span
          key={index}
          initial={{ opacity: 0, x: -5 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2 }}
          className="mr-1 inline-block"
        >
          <p className="text-[#4B5563]">{word}</p>
        </motion.span>
      ))}
    </div>
  );
}
